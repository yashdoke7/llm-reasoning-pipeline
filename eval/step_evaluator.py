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


_BACKTRACK_SYSTEM = """You are a reasoning error analyst. A model solved a problem but got the WRONG answer.
Given the correct answer, identify exactly which reasoning step FIRST went wrong.

Respond ONLY with valid JSON:
{
  "first_error_step": <integer step number>,
  "error_type": "logic_error" | "hallucination" | "wrong_approach" | "calculation_error" | "missing_step",
  "explanation": "<what specifically went wrong in that step>"
}

Be precise. Identify the FIRST step where reasoning diverges from what would lead to the correct answer.
If the fundamental approach is wrong from the start, say step 1."""


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

_ANSWER_JUDGE_SYSTEM = """You are an answer correctness evaluator.

Given a ground truth answer and a model's predicted answer, determine if they are semantically equivalent — meaning they convey the same factual content, even if worded differently.

Respond ONLY with valid JSON:
{
  "correct": true | false,
  "reasoning": "<one sentence explaining your decision>"
}

Rules:
- "correct" = true if the prediction contains the key facts from the ground truth
- Minor wording differences, synonyms, or extra context = still correct
- Missing key facts, wrong numbers/names/dates, contradictory statements = incorrect
- Do NOT penalize for additional correct information"""


def _check_answer_correctness(
    predicted: str,
    ground_truth: str,
    category: str,
    judge_client=None,
    judge_model: str = None,
) -> Optional[bool]:
    """
    Final answer comparison.

    For arithmetic: numeric comparison (objective, no LLM needed).
    For other categories: tries string heuristics first, then falls back
    to LLM-based semantic checking if a judge client is provided.

    Args:
        predicted:    Model's final answer string
        ground_truth: Reference answer
        category:     Task category (determines checking strategy)
        judge_client: Optional LLM client for semantic checking fallback
        judge_model:  Model to use for semantic checking

    Returns:
        True / False / None (None = cannot determine)
    """
    if not predicted or not ground_truth:
        return None

    pred = predicted.strip().lower()
    truth = ground_truth.strip().lower()

    # Exact match — always works
    if pred == truth:
        return True

    import re

    def extract_numbers(s: str) -> list[str]:
        return re.findall(r"-?\d+(?:\.\d+)?", s)

    # ── Arithmetic: numeric match (objective, trust this completely) ──
    if category == "multistep_arithmetic":
        pred_nums = extract_numbers(pred)
        truth_nums = extract_numbers(truth)
        if pred_nums and truth_nums:
            try:
                if abs(float(pred_nums[-1]) - float(truth_nums[-1])) < 1e-6:
                    return True
            except ValueError:
                pass
        return False  # for arithmetic, if number doesn't match → wrong

    # ── Non-arithmetic: try heuristics, then LLM fallback ────────────

    # Fast heuristic checks first (no API call)
    heuristic_result = _heuristic_check(pred, truth, category)
    if heuristic_result is True:
        return True  # confident enough to accept

    # If judge available, use it for semantic verification
    # (critical for non-arithmetic where wording varies significantly)
    if judge_client and judge_model:
        return _llm_answer_check(predicted, ground_truth, category, judge_client, judge_model)

    # No judge available — fall back to heuristic result (may be None or False)
    return heuristic_result


def _heuristic_check(pred: str, truth: str, category: str) -> Optional[bool]:
    """Fast string-based checks. Returns True (confident match), False (confident mismatch), or None (uncertain)."""
    import re

    if category == "factual_consistency":
        # 60% content word overlap — reasonable for short factual answers
        truth_words = set(truth.split())
        pred_words = set(pred.split())
        overlap = truth_words & pred_words
        if truth_words and len(overlap) / len(truth_words) > 0.6:
            return True
        if truth_words and len(overlap) / len(truth_words) < 0.1:
            return False
        return None  # uncertain — let LLM decide

    if category == "tool_use_planning":
        # Check tool names appear in predicted steps
        truth_tools = re.findall(r"(?:call\s+)?(\w+)\s+with", truth, re.I)
        if truth_tools:
            matched = sum(1 for t in truth_tools if t.lower() in pred)
            ratio = matched / len(truth_tools)
            if ratio >= 0.7:
                return True
            if ratio < 0.3:
                return False
        return None

    if category == "causal_counterfactual":
        # Content word overlap (stricter — counterfactuals need key concepts)
        stop = {'the', 'a', 'an', 'is', 'was', 'were', 'would', 'have', 'has',
                'had', 'been', 'be', 'to', 'of', 'and', 'in', 'that', 'it',
                'not', 'this', 'if', 'then', 'so', 'but', 'or', 'for', 'with'}
        truth_content = set(re.findall(r'\w+', truth)) - stop
        pred_content = set(re.findall(r'\w+', pred)) - stop
        if truth_content:
            overlap = len(truth_content & pred_content) / len(truth_content)
            if overlap >= 0.6:
                return True
            if overlap < 0.15:
                return False
        return None

    return None


def _llm_answer_check(
    predicted: str,
    ground_truth: str,
    category: str,
    client,
    model: str,
) -> Optional[bool]:
    """
    Use a judge LLM to semantically verify answer correctness.
    This eliminates false negatives from keyword mismatch and false positives
    from coincidental word overlap.
    """
    try:
        prompt = (
            f"Category: {category}\n\n"
            f"Ground truth answer: {ground_truth}\n\n"
            f"Model's predicted answer: {predicted}\n\n"
            f"Is the prediction semantically equivalent to the ground truth?"
        )
        raw = client.complete_json(
            model=model,
            user_prompt=prompt,
            system_prompt=_ANSWER_JUDGE_SYSTEM,
            temperature=0.0,
            max_tokens=150,
        )
        return bool(raw.get("correct", False))
    except Exception as e:
        logger.warning(f"LLM answer check failed: {e}")
        return None



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
        judge_client=None,
        judge_model: str = None,
    ) -> None:
        self._client = client
        self._model = evaluator_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._skip_single_step = skip_single_step
        # Separate judge for answer verification (eliminates self-grading bias)
        # If None, falls back to heuristic string matching
        self._judge_client = judge_client
        self._judge_model = judge_model

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
        # For tool_use, use full plan steps as "answer" (not just "Plan complete.")
        if task.category == "tool_use_planning" and task.steps:
            predicted_for_match = " ".join(s.text for s in task.steps)
        else:
            predicted_for_match = task.final_answer
        result.final_answer_correct = _check_answer_correctness(
            predicted_for_match,
            task.ground_truth,
            task.category,
            judge_client=self._judge_client,
            judge_model=self._judge_model,
        )

        # Error propagation: step failed AND final answer wrong
        if result.first_failure_step is not None and result.final_answer_correct is False:
            result.error_propagated = True

        # Ground-truth-aware backtrack: if answer is WRONG but all steps VALID,
        # run a second pass to find where reasoning actually diverged
        if (
            result.final_answer_correct is False
            and result.num_invalid == 0
            and task.ground_truth
            and len(task.steps) >= 2
        ):
            self._ground_truth_backtrack(result, task)

        return result

    def check_answer_correctness(
        self,
        predicted: str,
        ground_truth: str,
        category: str,
    ) -> Optional[bool]:
        """Public wrapper for final-answer equivalence checks."""
        return _check_answer_correctness(
            predicted,
            ground_truth,
            category,
            judge_client=self._judge_client,
            judge_model=self._judge_model,
        )

    def _ground_truth_backtrack(
        self,
        result: TaskEvaluation,
        task: DecomposedTask,
    ) -> None:
        """
        Second-pass evaluation: when all steps were marked VALID but the
        final answer is wrong, use ground truth to identify where reasoning
        actually diverged. This is the key insight — anchor evaluation in
        the known-correct answer.
        """
        steps_text = "\n".join(f"Step {s.index}: {s.text}" for s in task.steps)
        prompt = (
            f"A model was asked to solve this problem:\n\n"
            f"{task.prompt[:500]}\n\n"
            f"The CORRECT answer is: {task.ground_truth}\n"
            f"The model's WRONG answer is: {task.final_answer}\n\n"
            f"The model's reasoning steps:\n{steps_text}\n\n"
            f"Identify which step FIRST introduces an error or diverges from "
            f"the reasoning that would lead to the correct answer."
        )

        try:
            raw = self._client.complete_json(
                model=self._model,
                user_prompt=prompt,
                system_prompt=_BACKTRACK_SYSTEM,
                temperature=0.05,
                max_tokens=256,
            )

            error_step = int(raw.get("first_error_step", 1))
            error_type = str(raw.get("error_type", "reasoning_divergence"))
            explanation = raw.get("explanation", "Backtrack: divergence from correct answer")

            # Update the identified step
            for se in result.step_evaluations:
                if se.step_index == error_step:
                    se.verdict = StepVerdict.INVALID
                    se.error_type = error_type
                    se.explanation = f"[Backtrack] {explanation}"
                    se.confidence = 0.8
                    break

            # Recalculate aggregates
            result.num_invalid = sum(
                1 for se in result.step_evaluations
                if se.verdict == StepVerdict.INVALID
            )
            result.first_failure_step = error_step
            result.error_propagated = True

            logger.debug(
                f"[{result.task_id}] Backtrack found error at step {error_step}: "
                f"{error_type}"
            )
        except Exception as e:
            logger.warning(f"[{result.task_id}] Backtrack eval error: {e}")

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
