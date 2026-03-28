"""
data_loaders/physics_loader.py
Loader for physics reasoning datasets stored as JSONL.

Expected JSONL fields per line:
  - question OR problem OR prompt (required)
  - ground_truth OR final_answer OR answer (optional but recommended)
  - id (optional; auto-generated if absent)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PATH = os.path.join(_HERE, "data", "physics_eval.jsonl")


@dataclass
class PhysicsSample:
    id: str
    question: str
    ground_truth: str = ""
    category: str = "physics_reasoning"


def _extract_question(row: dict) -> str:
    return (
        row.get("question")
        or row.get("problem")
        or row.get("prompt")
        or ""
    ).strip()


def _extract_ground_truth(row: dict) -> str:
    return (
        row.get("ground_truth")
        or row.get("final_answer")
        or row.get("answer")
        or ""
    ).strip()


def load_physics(
    path: Optional[str] = None,
    max_samples: Optional[int] = 60,
) -> list[PhysicsSample]:
    resolved = path or _DEFAULT_PATH
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"Physics dataset not found: {resolved}. "
            "Create it first (see docs/EXECUTION_PLAYBOOK.md)."
        )

    samples: list[PhysicsSample] = []
    with open(resolved, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"Skipping malformed JSONL line {i} in {resolved}")
                continue

            question = _extract_question(row)
            if not question:
                logger.debug(f"Skipping physics row {i}: missing question/problem/prompt")
                continue

            samples.append(
                PhysicsSample(
                    id=row.get("id", f"physics_{i:04d}"),
                    question=question,
                    ground_truth=_extract_ground_truth(row),
                )
            )

    logger.info(f"Loaded {len(samples)} physics reasoning samples from {resolved}")
    return samples


def sample_to_dict(s: PhysicsSample) -> dict:
    return {
        "id": s.id,
        "category": s.category,
        "question": s.question,
        "ground_truth": s.ground_truth,
    }

