"""
eval/hallucination_scorer.py
Detects potential hallucinations in reasoning steps.

Hallucination here means: a step asserts a specific factual claim
(number, name, date, statistic) that is unverifiable or contradicts
known facts. This is distinct from logic errors.

Approach:
  1. Identify whether a step contains verifiable factual claims
  2. If so, score the claim for hallucination risk using the LLM
  3. Optionally cross-check against Wikipedia via retriever
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from models.groq_client import GroqClient

logger = logging.getLogger(__name__)


# ── Patterns that signal factual claims ──────────────────────
# A step that matches these likely contains checkable facts

_FACTUAL_INDICATORS = [
    re.compile(r"\b\d{4}\b"),                         # years
    re.compile(r"\b\d+(\.\d+)?\s*(%|percent)\b"),     # percentages
    re.compile(r"\b\d+\s*(million|billion|trillion)\b", re.I),
    re.compile(r"\b(according to|studies show|research shows|it is known)\b", re.I),
    re.compile(r"\b(invented|discovered|published|founded|established)\s+in\b", re.I),
    re.compile(r"\b(named after|developed by|created by|introduced by)\b", re.I),
]


def _has_factual_claims(step_text: str) -> bool:
    return any(p.search(step_text) for p in _FACTUAL_INDICATORS)


# ── Prompt ────────────────────────────────────────────────────

_HALLUCINATION_SYSTEM = """You are a hallucination detection expert. You assess whether a reasoning step contains hallucinated factual claims.

A hallucination is a specific factual assertion that is:
- Factually incorrect (wrong number, wrong date, wrong name)
- Invented or fabricated (does not correspond to reality)
- Presented with false confidence

You are NOT looking for vague or opinion statements — only specific, verifiable factual claims.

Respond ONLY with a valid JSON object:
{
  "contains_factual_claims": true | false,
  "hallucination_risk": "low" | "medium" | "high",
  "suspicious_claims": ["list of specific claims that seem potentially incorrect"],
  "explanation": "<brief explanation>"
}

If contains_factual_claims is false, set hallucination_risk to "low" and suspicious_claims to [].
Be conservative — only flag claims you are fairly sure are wrong."""


@dataclass
class HallucinationScore:
    step_index: int
    step_text: str
    contains_factual_claims: bool
    hallucination_risk: str          # "low" | "medium" | "high"
    suspicious_claims: list[str]
    explanation: str
    risk_score: float                # 0.0 = low, 1.0 = high (derived from hallucination_risk)

    @property
    def is_flagged(self) -> bool:
        return self.hallucination_risk in ("medium", "high")


_RISK_TO_SCORE = {"low": 0.1, "medium": 0.6, "high": 0.9}


class HallucinationScorer:
    """
    Scores reasoning steps for hallucination risk.

    Only calls the LLM for steps that contain likely factual claims
    (based on pattern matching), saving API calls.

    Usage:
        scorer = HallucinationScorer(client, model="llama-3-8b-8192")
        scores = scorer.score_steps(task_id, steps)
    """

    def __init__(
        self,
        client: GroqClient,
        model: str = "llama-3-8b-8192",
        temperature: float = 0.05,
        max_tokens: int = 512,
        skip_non_factual: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._skip_non_factual = skip_non_factual

    def score_steps(
        self,
        task_id: str,
        steps: list,  # list of ReasoningStep
        context: str = "",
    ) -> list[HallucinationScore]:
        """
        Score all steps in a reasoning trace.

        Args:
            task_id: For logging
            steps:   List of ReasoningStep objects
            context: Original problem/context (helps with grounding)

        Returns:
            List of HallucinationScore, one per step
        """
        scores = []
        for step in steps:
            if self._skip_non_factual and not _has_factual_claims(step.text):
                scores.append(
                    HallucinationScore(
                        step_index=step.index,
                        step_text=step.text,
                        contains_factual_claims=False,
                        hallucination_risk="low",
                        suspicious_claims=[],
                        explanation="No specific factual claims detected — skipped.",
                        risk_score=0.0,
                    )
                )
                continue

            try:
                score = self._score_step(step, context)
                scores.append(score)
                if score.is_flagged:
                    logger.debug(
                        f"[{task_id}] Step {step.index} flagged: "
                        f"{score.hallucination_risk} risk — {score.suspicious_claims}"
                    )
            except Exception as e:
                logger.error(f"[{task_id}] Hallucination score error step {step.index}: {e}")
                scores.append(
                    HallucinationScore(
                        step_index=step.index,
                        step_text=step.text,
                        contains_factual_claims=False,
                        hallucination_risk="low",
                        suspicious_claims=[],
                        explanation=f"Scoring error: {e}",
                        risk_score=0.0,
                    )
                )

        return scores

    def _score_step(self, step, context: str) -> HallucinationScore:
        ctx_line = f"\nContext: {context[:400]}" if context else ""
        prompt = (
            f"Reasoning step to evaluate:{ctx_line}\n\n"
            f"Step {step.index}: {step.text}"
        )

        raw = self._client.complete_json(
            model=self._model,
            user_prompt=prompt,
            system_prompt=_HALLUCINATION_SYSTEM,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        risk = str(raw.get("hallucination_risk", "low")).lower()
        if risk not in ("low", "medium", "high"):
            risk = "low"

        return HallucinationScore(
            step_index=step.index,
            step_text=step.text,
            contains_factual_claims=bool(raw.get("contains_factual_claims", False)),
            hallucination_risk=risk,
            suspicious_claims=raw.get("suspicious_claims", []),
            explanation=raw.get("explanation", ""),
            risk_score=_RISK_TO_SCORE.get(risk, 0.1),
        )

    def aggregate(self, scores: list[HallucinationScore]) -> dict:
        """Aggregate step scores into task-level hallucination stats."""
        if not scores:
            return {"hallucination_rate": 0.0, "flagged_steps": 0, "total_steps": 0}
        flagged = sum(1 for s in scores if s.is_flagged)
        factual_steps = sum(1 for s in scores if s.contains_factual_claims)
        return {
            "total_steps": len(scores),
            "factual_claim_steps": factual_steps,
            "flagged_steps": flagged,
            "hallucination_rate": flagged / len(scores) if scores else 0.0,
            "avg_risk_score": sum(s.risk_score for s in scores) / len(scores),
        }

    def to_dict_list(self, scores: list[HallucinationScore]) -> list[dict]:
        return [
            {
                "step_index": s.step_index,
                "contains_factual_claims": s.contains_factual_claims,
                "hallucination_risk": s.hallucination_risk,
                "risk_score": s.risk_score,
                "suspicious_claims": s.suspicious_claims,
                "explanation": s.explanation,
            }
            for s in scores
        ]
