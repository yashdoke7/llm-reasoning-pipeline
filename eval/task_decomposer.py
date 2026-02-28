"""
eval/task_decomposer.py
Converts dataset samples into structured chain-of-thought prompts
and parses model outputs into discrete, numbered reasoning steps.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────

@dataclass
class ReasoningStep:
    index: int            # 1-based
    text: str             # full text of the step
    is_final: bool = False  # True for the final answer/conclusion step


@dataclass
class DecomposedTask:
    task_id: str
    category: str
    prompt: str                        # full prompt sent to LLM
    raw_response: str = ""             # raw LLM output
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    ground_truth: str = ""
    parse_error: Optional[str] = None


# ── System prompts per category ───────────────────────────────

_SYSTEM_PROMPTS = {
    "multistep_arithmetic": (
        "You are a precise mathematical reasoner. "
        "Solve the problem step by step. "
        "Number each step as 'Step 1:', 'Step 2:', etc. "
        "Each step must contain exactly one logical operation or inference. "
        "End your response with 'Final Answer: <number>' on its own line."
    ),
    "factual_consistency": (
        "You are a careful analytical reasoner. "
        "Answer each question by reasoning step by step, "
        "explicitly referencing facts from the provided context. "
        "Number each reasoning step as 'Step 1:', 'Step 2:', etc. "
        "End with 'Final Answer: <your consolidated answers>' on its own line."
    ),
    "tool_use_planning": (
        "You are a systematic planning agent. "
        "Given a goal and a set of available tools, produce a step-by-step "
        "execution plan. Each step must call exactly one tool with explicit "
        "parameter values. Number steps as 'Step 1:', 'Step 2:', etc. "
        "End with 'Final Answer: Plan complete.' on its own line."
    ),
    "causal_counterfactual": (
        "You are a careful causal and counterfactual reasoner. "
        "Reason step by step about what would be different given the "
        "hypothetical premise. Number each inference as 'Step 1:', 'Step 2:', etc. "
        "Ground each step in facts or logical necessity. "
        "End with 'Final Answer: <your conclusion>' on its own line."
    ),
}

_DEFAULT_SYSTEM = (
    "Reason step by step. Number each step as 'Step 1:', 'Step 2:', etc. "
    "End with 'Final Answer: <answer>' on its own line."
)


# ── Prompt formatters per category ───────────────────────────

def _format_arithmetic(question: str, **_) -> str:
    return f"Problem: {question}"


def _format_factual(context: str, questions: list[str], **_) -> str:
    q_block = "\n".join(f"Q{i+1}: {q}" for i, q in enumerate(questions))
    return (
        f"Read the following information carefully:\n\n"
        f"{context}\n\n"
        f"Answer each of the following questions by reasoning step by step, "
        f"citing specific facts from the text above:\n\n{q_block}"
    )


def _format_tool_use(goal: str, available_tools: list[dict], constraints: list[str], **_) -> str:
    tool_lines = []
    for t in available_tools:
        params = ", ".join(t.get("params", []))
        tool_lines.append(f"  - {t['name']}({params}): {t.get('description', '')}")
    tools_str = "\n".join(tool_lines)
    constraint_str = ""
    if constraints:
        constraint_str = "\n\nConstraints (must not violate):\n" + "\n".join(f"  - {c}" for c in constraints)
    return (
        f"Goal: {goal}\n\n"
        f"Available tools:\n{tools_str}"
        f"{constraint_str}\n\n"
        f"Produce a step-by-step execution plan."
    )


def _format_counterfactual(premise: str, question: str, **_) -> str:
    return (
        f"Consider this hypothetical premise:\n\n"
        f"\"{premise}\"\n\n"
        f"Question: {question}\n\n"
        f"Reason step by step about what would follow from this premise."
    )


_FORMATTERS = {
    "multistep_arithmetic": _format_arithmetic,
    "factual_consistency": _format_factual,
    "tool_use_planning": _format_tool_use,
    "causal_counterfactual": _format_counterfactual,
}


# ── Parser ────────────────────────────────────────────────────

# Matches "Step 1:", "Step 1.", "1.", "1)" etc.
_STEP_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Step\s+)?(\d+)[.:)]\s*(.+?)(?=\n\s*(?:Step\s+)?\d+[.:)]|\n\s*Final Answer|$)",
    re.DOTALL | re.IGNORECASE,
)
_FINAL_ANSWER_PATTERN = re.compile(
    r"Final\s+Answer\s*[:\-]?\s*(.+?)$",
    re.IGNORECASE | re.DOTALL,
)


def parse_reasoning_trace(raw: str) -> tuple[list[ReasoningStep], str]:
    """
    Parse a raw LLM reasoning trace into discrete steps and a final answer.

    Returns:
        (steps, final_answer)
        steps: list of ReasoningStep in order
        final_answer: extracted final answer string (empty if not found)
    """
    # Extract final answer first
    final_answer = ""
    final_match = _FINAL_ANSWER_PATTERN.search(raw)
    if final_match:
        final_answer = final_match.group(1).strip()
        # Remove final answer block for step parsing
        text_for_steps = raw[: final_match.start()]
    else:
        text_for_steps = raw

    # Extract numbered steps
    steps = []
    for match in _STEP_PATTERN.finditer(text_for_steps):
        idx = int(match.group(1))
        text = match.group(2).strip()
        if text:
            steps.append(ReasoningStep(index=idx, text=text))

    # If no numbered steps found, try splitting by newlines as fallback
    if not steps:
        lines = [l.strip() for l in text_for_steps.split("\n") if l.strip()]
        # Filter out very short lines (likely headers)
        lines = [l for l in lines if len(l) > 20]
        steps = [ReasoningStep(index=i + 1, text=l) for i, l in enumerate(lines)]
        logger.debug("Step parsing fallback: split by newlines")

    # Mark the last step as final if no explicit Final Answer
    if steps and not final_answer:
        steps[-1].is_final = True
        final_answer = steps[-1].text

    return steps, final_answer


# ── Main decomposer ───────────────────────────────────────────

class TaskDecomposer:
    """
    Converts dataset samples into prompts and parses LLM responses into steps.

    Usage:
        decomposer = TaskDecomposer()
        task = decomposer.build("task_001", "multistep_arithmetic",
                                question="If a train...", ground_truth="42")
        # Later, after getting LLM response:
        task = decomposer.parse_response(task, llm_raw_output)
    """

    def get_system_prompt(self, category: str) -> str:
        return _SYSTEM_PROMPTS.get(category, _DEFAULT_SYSTEM)

    def build_prompt(self, category: str, **task_fields) -> str:
        """
        Build the user-facing prompt from task fields.

        task_fields depends on category:
          multistep_arithmetic: question
          factual_consistency:  context, questions (list)
          tool_use_planning:    goal, available_tools, constraints
          causal_counterfactual: premise, question
        """
        formatter = _FORMATTERS.get(category)
        if formatter is None:
            raise ValueError(f"Unknown category: {category}")
        return formatter(**task_fields)

    def build(
        self,
        task_id: str,
        category: str,
        ground_truth: str = "",
        **task_fields,
    ) -> DecomposedTask:
        """Create a DecomposedTask ready for LLM completion."""
        prompt = self.build_prompt(category, **task_fields)
        return DecomposedTask(
            task_id=task_id,
            category=category,
            prompt=prompt,
            ground_truth=ground_truth,
        )

    def parse_response(
        self,
        task: DecomposedTask,
        raw_response: str,
    ) -> DecomposedTask:
        """
        Parse the raw LLM response into structured steps and final answer.
        Mutates and returns the task.
        """
        task.raw_response = raw_response
        try:
            steps, final_answer = parse_reasoning_trace(raw_response)
            task.steps = steps
            task.final_answer = final_answer
            if len(steps) < 1:
                task.parse_error = "No reasoning steps found in response"
                logger.warning(f"[{task.task_id}] No steps found")
        except Exception as e:
            task.parse_error = str(e)
            logger.error(f"[{task.task_id}] Parse error: {e}")
        return task

    def to_dict(self, task: DecomposedTask) -> dict:
        return {
            "task_id": task.task_id,
            "category": task.category,
            "prompt": task.prompt,
            "raw_response": task.raw_response,
            "steps": [{"index": s.index, "text": s.text, "is_final": s.is_final} for s in task.steps],
            "final_answer": task.final_answer,
            "ground_truth": task.ground_truth,
            "parse_error": task.parse_error,
            "num_steps": len(task.steps),
        }
