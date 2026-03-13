"""
datasets/crass_loader.py
Load CRASS (Counterfactual Reasoning Assessment) dataset.

DOWNLOAD INSTRUCTIONS:
  1. Go to: https://github.com/apergo-ai/CRASS
  2. Download: crass_dataset.csv  (in the /data folder of the repo)
  3. Place it at: datasets/data/crass_dataset.csv

CSV columns (CRASS v1):
  premise, question, correct_answer, distractor_1, distractor_2, distractor_3

If the file is missing, this loader auto-generates a small synthetic
counterfactual dataset so the pipeline still runs end-to-end.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PATH = os.path.join(_HERE, "data", "crass_dataset.csv")


@dataclass
class CRASSSample:
    id: str
    premise: str
    question: str
    correct_answer: str
    distractors: list[str]
    category: str = "causal_counterfactual"


# ── Synthetic fallback ────────────────────────────────────────

_SYNTHETIC_SAMPLES = [
    {
        "premise": "If water were not composed of hydrogen and oxygen, it would not be liquid at room temperature.",
        "question": "What would happen to Earth's oceans if water lacked its hydrogen-oxygen composition?",
        "correct_answer": "The oceans would not exist as liquid bodies of water.",
        "distractors": ["The oceans would become larger.", "The oceans would freeze permanently.", "Nothing would change."],
    },
    {
        "premise": "If the Internet had never been invented, global communication would still rely on physical mail and telephone.",
        "question": "How would scientific research be different without the Internet?",
        "correct_answer": "Researchers would collaborate much more slowly, relying on physical journals and postal correspondence.",
        "distractors": ["Research would proceed at the same pace.", "Science would have advanced faster.", "All research would have stopped."],
    },
    {
        "premise": "If photosynthesis had never evolved, plants would not produce oxygen.",
        "question": "What would the atmosphere be like if photosynthesis had never evolved?",
        "correct_answer": "The atmosphere would have very little free oxygen, making it hostile to most animal life.",
        "distractors": ["The atmosphere would have more oxygen.", "The atmosphere would be the same as today.", "All gases would disappear."],
    },
    {
        "premise": "If Newton had not formulated his laws of motion, classical mechanics would have developed differently.",
        "question": "How would engineering in the 18th century have differed without Newton's laws?",
        "correct_answer": "Engineers would have lacked a mathematical framework for predicting motion, slowing industrial development.",
        "distractors": ["Engineering would have advanced faster.", "Nothing would be different.", "All machines would have failed to work."],
    },
    {
        "premise": "If the printing press had not been invented, literacy rates in Europe would have remained low.",
        "question": "How would the Protestant Reformation have been affected without the printing press?",
        "correct_answer": "The Reformation would have spread far more slowly without the ability to mass-produce pamphlets and Bibles.",
        "distractors": ["The Reformation would have happened faster.", "There would have been no religious conflict.", "Books would still have been widely available."],
    },
    {
        "premise": "If antibiotics had never been discovered, bacterial infections would remain the leading cause of death.",
        "question": "How would surgery be different in a world without antibiotics?",
        "correct_answer": "Most surgical operations would be too risky due to the high probability of fatal post-operative bacterial infections.",
        "distractors": ["Surgery would be safer.", "Surgeons would use better techniques.", "Hospitals would be unnecessary."],
    },
    {
        "premise": "If electricity had never been harnessed for practical use, industrial production would depend entirely on steam and manual labor.",
        "question": "How would cities be different in a world without electrical power?",
        "correct_answer": "Cities would lack electric lighting, powered transportation, and most modern appliances, dramatically altering daily life.",
        "distractors": ["Cities would be identical.", "Cities would be cleaner.", "Buildings would be taller."],
    },
    {
        "premise": "If the ozone layer did not exist, ultraviolet radiation reaching Earth's surface would be far more intense.",
        "question": "How would agriculture be affected without the ozone layer?",
        "correct_answer": "Crops would suffer severe DNA damage from UV radiation, dramatically reducing food production.",
        "distractors": ["Crops would grow faster.", "Agriculture would be unaffected.", "Only root vegetables would survive."],
    },
    {
        "premise": "If humans had never domesticated animals, early civilizations would lack draft power for farming and transport.",
        "question": "How would the development of cities have differed without animal domestication?",
        "correct_answer": "Without animal labor for agriculture and transport, surpluses and specialization that enable cities would have developed much more slowly.",
        "distractors": ["Cities would have developed faster.", "Animals are irrelevant to city formation.", "Only coastal cities would exist."],
    },
    {
        "premise": "If the compass had not been invented, maritime navigation would be limited to coastal routes.",
        "question": "How would the Age of Exploration have differed without the compass?",
        "correct_answer": "European explorers could not have safely crossed open oceans, delaying or preventing contact between continents.",
        "distractors": ["Exploration would have happened even faster.", "Ships would have used stars only, with equal accuracy.", "The Americas would still have been discovered at the same time."],
    },
]


def load_crass(
    path: Optional[str] = None,
    max_samples: Optional[int] = 50,
) -> list[CRASSSample]:
    """
    Load CRASS counterfactual reasoning samples.

    Priority:
      1. Generated JSONL (from data_loaders/generate_eval_datasets.py) — best option
      2. Real CRASS CSV (download from https://github.com/apergo-ai/CRASS)
      3. Synthetic fallback (10 hardcoded samples) — last resort

    Args:
        path:        Path to dataset file (.jsonl or .csv)
        max_samples: Limit to N samples

    Returns:
        List of CRASSSample
    """
    resolved = path or _DEFAULT_PATH

    if not os.path.exists(resolved):
        # Also check for generated JSONL in default location
        generated_path = os.path.join(_HERE, "data", "generated_counterfactual.jsonl")
        if os.path.exists(generated_path):
            logger.info(f"Loading generated counterfactual dataset from {generated_path}")
            return _load_jsonl(generated_path, max_samples)
        logger.warning(
            f"Counterfactual dataset not found at '{resolved}'. "
            "Using synthetic fallback (10 samples). "
            "Generate better data: python data_loaders/generate_eval_datasets.py --categories causal_counterfactual"
        )
        return _load_synthetic(max_samples)

    # JSONL format (generated)
    if resolved.endswith(".jsonl"):
        logger.info(f"Loading generated counterfactual JSONL from {resolved}")
        return _load_jsonl(resolved, max_samples)

    logger.info(f"Loading CRASS CSV from {resolved}")
    return _load_csv(resolved, max_samples)


def _load_jsonl(path: str, max_samples: Optional[int]) -> list[CRASSSample]:
    """Load generated JSONL counterfactual dataset."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and len(samples) >= max_samples:
                break
            try:
                row = json.loads(line.strip())
                samples.append(CRASSSample(
                    id=row.get("id", f"crass_gen_{i:04d}"),
                    premise=row["premise"],
                    question=row["question"],
                    correct_answer=row["correct_answer"],
                    distractors=row.get("distractors", []),
                ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Skipping malformed line {i}: {e}")
    logger.info(f"Loaded {len(samples)} counterfactual samples from JSONL")
    return samples


def _load_csv(path: str, max_samples: Optional[int]) -> list[CRASSSample]:
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        # Detect delimiter: try semicolon first (CRASS default), then comma
        first_line = f.readline()
        f.seek(0)
        delimiter = ";" if ";" in first_line else ","

        reader = csv.DictReader(f, delimiter=delimiter)
        headers = reader.fieldnames or []
        logger.info(f"CRASS CSV headers (delimiter='{delimiter}'): {headers}")
        
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            
            # Try every possible column name variant
            premise = (row.get("premise") or row.get("context") or 
                      row.get("Premise") or row.get("Context") or "")
            question = (row.get("question") or row.get("counterfactual") or 
                       row.get("Question") or row.get("Counterfactual") or 
                       row.get("QCC") or row.get("qcc") or "")
            correct = (row.get("correct_answer") or row.get("answer") or
                      row.get("Correct_answer") or row.get("Answer") or
                      row.get("correct") or row.get("Correct") or row.get("CorrectAnswer") or "")
            distractors = [
                row.get("Answer1", "") or row.get("Distractor_1", ""),
                row.get("Answer2", "") or row.get("Distractor_2", ""),
                row.get("PossibleAnswer3", "") or row.get("Distractor_3", ""),
            ]
            distractors = [d for d in distractors if d]

            if not premise or not question or not correct:
                if i == 0:
                    logger.warning(f"First row could not be parsed. Row keys: {list(row.keys())}")
                continue

            samples.append(
                CRASSSample(
                    id=f"crass_{i:04d}",
                    premise=premise.strip(),
                    question=question.strip(),
                    correct_answer=correct.strip(),
                    distractors=distractors,
                )
            )

    logger.info(f"Loaded {len(samples)} CRASS samples")
    return samples


def _load_synthetic(max_samples: Optional[int]) -> list[CRASSSample]:
    data = _SYNTHETIC_SAMPLES
    if max_samples:
        data = data[:max_samples]
    samples = []
    for i, row in enumerate(data):
        samples.append(
            CRASSSample(
                id=f"crass_synthetic_{i:04d}",
                premise=row["premise"],
                question=row["question"],
                correct_answer=row["correct_answer"],
                distractors=row["distractors"],
            )
        )
    logger.info(f"Returning {len(samples)} synthetic CRASS samples")
    return samples


def sample_to_dict(s: CRASSSample) -> dict:
    return {
        "id": s.id,
        "category": s.category,
        "premise": s.premise,
        "question": s.question,
        "correct_answer": s.correct_answer,
        "distractors": s.distractors,
    }
