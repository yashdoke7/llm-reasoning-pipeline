"""
mitigation/regrounder.py
Re-grounds a failing reasoning trace by injecting retrieved context
and re-generating the problematic step or the full answer.

Three intervention modes:
  1. STEP: Re-generate only the failing step with context injected
  2. FULL: Re-generate the full answer with context injected
  3. REPROMPT: Re-ask without context (control condition for ablation)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from models.groq_client import GroqClient
from mitigation.retriever import WikipediaRetriever, RetrievedDoc
from eval.task_decomposer import DecomposedTask, TaskDecomposer, parse_reasoning_trace

logger = logging.getLogger(__name__)


class InterventionMode(str, Enum):
    STEP = "step"
    FULL = "full"
    REPROMPT = "reprompt"      # no context — ablation baseline


@dataclass
class RegoundResult:
    task_id: str
    mode: InterventionMode
    original_final_answer: str
    regrounded_answer: str
    retrieved_docs: list[RetrievedDoc]
    raw_response: str
    success: bool
    error: Optional[str] = None


_REGROUND_SYSTEM = """You are a careful reasoning assistant. You have been provided with relevant reference information.
Use this information to correct or ground your reasoning. Be precise and cite facts from the provided sources."""


class Regrounder:
    """
    Applies RAG re-grounding to failing or hallucinating reasoning tasks.

    Compares three conditions:
      - no_intervention:  original response (already computed)
      - reprompt:         retry without context (control)
      - rag_reground:     retry with retrieved Wikipedia context

    Usage:
        regrounder = Regrounder(client, retriever)
        result = regrounder.reground(task, suspicious_claims=["..."])
    """

    def __init__(
        self,
        client: GroqClient,
        retriever: WikipediaRetriever,
        model: Optional[str] = None,
        temperature: float = 0.05,
        max_tokens: int = 1024,
        max_attempts: int = 2,
    ) -> None:
        self._client = client
        self._retriever = retriever
        self._model = model   # if None, will be set per call
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_attempts = max_attempts
        self._decomposer = TaskDecomposer()

    def reground(
        self,
        task: DecomposedTask,
        model: str,
        suspicious_claims: Optional[list[str]] = None,
        mode: InterventionMode = InterventionMode.FULL,
    ) -> RegoundResult:
        """
        Re-ground a task's reasoning using retrieved context.

        Args:
            task:              The original DecomposedTask (already has steps/response)
            model:             Model to use for re-generation
            suspicious_claims: Specific claims to retrieve context for
            mode:              Intervention mode

        Returns:
            RegoundResult with the new answer
        """
        # Retrieve context
        docs: list[RetrievedDoc] = []
        if mode == InterventionMode.FULL or mode == InterventionMode.STEP:
            if suspicious_claims:
                docs = self._retriever.retrieve_for_claims(suspicious_claims)
            else:
                # Retrieve based on first 150 chars of the problem
                docs = self._retriever.retrieve(task.prompt[:150])

        try:
            if mode == InterventionMode.REPROMPT:
                return self._reprompt(task, model)
            else:
                return self._rag_reground(task, model, docs, mode)
        except Exception as e:
            logger.error(f"[{task.task_id}] Reground error: {e}")
            return RegoundResult(
                task_id=task.task_id,
                mode=mode,
                original_final_answer=task.final_answer,
                regrounded_answer="",
                retrieved_docs=docs,
                raw_response="",
                success=False,
                error=str(e),
            )

    def _reprompt(self, task: DecomposedTask, model: str) -> RegoundResult:
        """Re-ask the original question without any additional context."""
        system = self._decomposer.get_system_prompt(task.category)
        raw = self._client.complete(
            model=model,
            user_prompt=task.prompt,
            system_prompt=system,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        _, final_answer = parse_reasoning_trace(raw.content)
        return RegoundResult(
            task_id=task.task_id,
            mode=InterventionMode.REPROMPT,
            original_final_answer=task.final_answer,
            regrounded_answer=final_answer,
            retrieved_docs=[],
            raw_response=raw.content,
            success=True,
        )

    def _rag_reground(
        self,
        task: DecomposedTask,
        model: str,
        docs: list[RetrievedDoc],
        mode: InterventionMode,
    ) -> RegoundResult:
        """Re-generate with retrieved context injected into prompt."""
        context_block = self._retriever.format_for_prompt(docs)

        if context_block:
            augmented_prompt = (
                f"Use the following reference information to inform your reasoning:\n\n"
                f"{context_block}\n\n"
                f"---\n\n"
                f"{task.prompt}"
            )
        else:
            augmented_prompt = task.prompt
            logger.debug(f"[{task.task_id}] No docs retrieved — reground without context")

        system = self._decomposer.get_system_prompt(task.category)
        raw = self._client.complete(
            model=model,
            user_prompt=augmented_prompt,
            system_prompt=system,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        _, final_answer = parse_reasoning_trace(raw.content)

        return RegoundResult(
            task_id=task.task_id,
            mode=mode,
            original_final_answer=task.final_answer,
            regrounded_answer=final_answer,
            retrieved_docs=docs,
            raw_response=raw.content,
            success=True,
        )

    def to_dict(self, result: RegoundResult) -> dict:
        return {
            "task_id": result.task_id,
            "mode": result.mode.value,
            "original_answer": result.original_final_answer,
            "regrounded_answer": result.regrounded_answer,
            "num_docs_retrieved": len(result.retrieved_docs),
            "retrieved_titles": [d.title for d in result.retrieved_docs],
            "success": result.success,
            "error": result.error,
        }
