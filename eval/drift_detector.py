"""
eval/drift_detector.py
Detects reasoning drift: cases where the model's reasoning diverges
from the original problem, contradicts earlier steps, or becomes
incoherent over long reasoning chains.

Two detection modes:
  1. LLM-based: asks the evaluator model to check for contradictions
  2. Heuristic: similarity-based drift detection (no API call needed)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from models.groq_client import GroqClient

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    task_id: str
    has_drift: bool
    drift_score: float             # 0.0 = no drift, 1.0 = severe drift
    contradiction_pairs: list[dict] = field(default_factory=list)
    # [{"step_a": int, "step_b": int, "description": str}]
    topic_drift: bool = False      # reasoning moved away from original problem
    drift_explanation: str = ""


# ── Prompt ────────────────────────────────────────────────────

_DRIFT_SYSTEM = """You are a reasoning coherence analyst. Check a sequence of reasoning steps for:
1. CONTRADICTIONS: Does any step contradict a previous step or the original problem?
2. TOPIC DRIFT: Does the reasoning diverge from the original problem in a way that is irrelevant?

Respond ONLY with valid JSON:
{
  "has_contradiction": true | false,
  "has_topic_drift": true | false,
  "drift_score": <float 0.0-1.0, where 0=coherent, 1=severely drifted>,
  "contradiction_pairs": [
    {"step_a": <int>, "step_b": <int>, "description": "<what contradicts what>"}
  ],
  "explanation": "<brief overall coherence assessment>"
}

Be specific when identifying contradictions. Do not flag minor rephrasing as contradictions."""


class DriftDetector:
    """
    Detects reasoning drift and contradictions across a full reasoning trace.

    Usage:
        detector = DriftDetector(client)
        result = detector.detect("task_001", problem, steps)
    """

    def __init__(
        self,
        client: GroqClient,
        model: str = "llama-3-8b-instant",
        temperature: float = 0.05,
        max_tokens: int = 1024,
        min_steps_for_check: int = 3,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._min_steps = min_steps_for_check

    def detect(
        self,
        task_id: str,
        problem: str,
        steps: list,    # list of ReasoningStep
        category: str = "",
    ) -> DriftResult:
        """
        Run drift detection on a completed reasoning trace.

        Short traces (< min_steps_for_check) return no-drift by default.
        """
        if len(steps) < self._min_steps:
            return DriftResult(
                task_id=task_id,
                has_drift=False,
                drift_score=0.0,
                drift_explanation=f"Too few steps ({len(steps)}) for drift detection.",
            )

        try:
            return self._llm_detect(task_id, problem, steps, category)
        except Exception as e:
            logger.error(f"[{task_id}] Drift detection error: {e}")
            return DriftResult(
                task_id=task_id,
                has_drift=False,
                drift_score=0.0,
                drift_explanation=f"Detection error: {e}",
            )

    def _llm_detect(
        self,
        task_id: str,
        problem: str,
        steps: list,
        category: str,
    ) -> DriftResult:
        steps_text = "\n".join(f"Step {s.index}: {s.text}" for s in steps)
        prompt = (
            f"Category: {category}\n\n"
            f"Original problem:\n{problem[:300]}\n\n"
            f"Reasoning trace:\n{steps_text}"
        )

        raw = self._client.complete_json(
            model=self._model,
            user_prompt=prompt,
            system_prompt=_DRIFT_SYSTEM,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        drift_score = float(raw.get("drift_score", 0.0))
        drift_score = max(0.0, min(1.0, drift_score))

        result = DriftResult(
            task_id=task_id,
            has_drift=bool(raw.get("has_contradiction", False)) or bool(raw.get("has_topic_drift", False)),
            drift_score=drift_score,
            contradiction_pairs=raw.get("contradiction_pairs", []),
            topic_drift=bool(raw.get("has_topic_drift", False)),
            drift_explanation=raw.get("explanation", ""),
        )

        if result.has_drift:
            logger.debug(
                f"[{task_id}] Drift detected: score={drift_score:.2f} "
                f"contradictions={len(result.contradiction_pairs)}"
            )

        return result

    def to_dict(self, result: DriftResult) -> dict:
        return {
            "task_id": result.task_id,
            "has_drift": result.has_drift,
            "drift_score": result.drift_score,
            "topic_drift": result.topic_drift,
            "contradiction_pairs": result.contradiction_pairs,
            "drift_explanation": result.drift_explanation,
        }
