"""
datasets/gsm8k_loader.py
Load and preprocess GSM8K math reasoning dataset.

GSM8K is auto-downloaded via HuggingFace datasets on first run.
No manual download needed.

Dataset format:
  question: str  — the math word problem
  answer:   str  — solution with step-by-step work + final answer (#### 42)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GSM8KSample:
    id: str
    question: str
    answer_text: str          # full solution text from dataset
    ground_truth: str         # extracted final numeric answer
    category: str = "multistep_arithmetic"


def _extract_final_answer(answer_text: str) -> str:
    """
    GSM8K answers end with '#### <number>'.
    Extract and return just the number as a string.
    """
    match = re.search(r"####\s*(.+)$", answer_text.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip().replace(",", "")
    # fallback: last number in text
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    return nums[-1] if nums else ""


def load_gsm8k(
    split: str = "test",
    max_samples: Optional[int] = 100,
    cache_dir: Optional[str] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[GSM8KSample]:
    """
    Load GSM8K samples from HuggingFace.

    Args:
        split:       "train" | "test"
        max_samples: Limit to N samples (None = all)
        cache_dir:   Local cache path for downloaded data
        shuffle:     Shuffle before subsetting
        seed:        Random seed for shuffling

    Returns:
        List of GSM8KSample
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info(f"Loading GSM8K ({split} split, max={max_samples})")
    ds = load_dataset(
        "gsm8k",
        "main",
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )

    if shuffle:
        ds = ds.shuffle(seed=seed)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples = []
    for i, row in enumerate(ds):
        samples.append(
            GSM8KSample(
                id=f"gsm8k_{split}_{i:04d}",
                question=row["question"].strip(),
                answer_text=row["answer"].strip(),
                ground_truth=_extract_final_answer(row["answer"]),
            )
        )

    logger.info(f"Loaded {len(samples)} GSM8K samples")
    return samples


def sample_to_dict(s: GSM8KSample) -> dict:
    return {
        "id": s.id,
        "category": s.category,
        "question": s.question,
        "ground_truth": s.ground_truth,
        "answer_text": s.answer_text,
    }
