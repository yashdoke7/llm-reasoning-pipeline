"""
datasets/factual_synthetic.py
Generate or load synthetic factual consistency tasks.

These tasks test whether LLMs can maintain factual accuracy across
long reasoning chains — specifically, whether they contradict facts
stated earlier in their own reasoning.

If a cached file exists at datasets/data/factual_synthetic.json, it is loaded.
Otherwise, tasks are generated programmatically from hardcoded fact-sets
(no LLM call needed — pure data construction).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_HERE, "data", "factual_synthetic.json")


@dataclass
class FactualSample:
    id: str
    context: str              # paragraph of facts to reason over
    questions: list[str]      # multiple questions requiring recall of context facts
    answers: list[str]        # ground truth answers (one per question)
    topic: str
    category: str = "factual_consistency"

    def to_full_task(self) -> str:
        """Format as a single multi-question reasoning task."""
        q_text = "\n".join(f"Q{i+1}: {q}" for i, q in enumerate(self.questions))
        return (
            f"Read the following information carefully:\n\n"
            f"{self.context}\n\n"
            f"Now answer each question by reasoning step by step, "
            f"referencing only the information provided above:\n\n"
            f"{q_text}"
        )


# ── Built-in fact sets ────────────────────────────────────────
# Each entry: topic, context paragraph, list of (question, answer) pairs

_FACT_SETS = [
    {
        "topic": "machine_learning",
        "context": (
            "The transformer architecture was introduced in 2017 in the paper "
            "'Attention Is All You Need' by Vaswani et al. It relies on a "
            "self-attention mechanism rather than recurrent connections. "
            "BERT, released by Google in 2018, is a transformer model trained "
            "with masked language modeling and next sentence prediction. "
            "GPT-2, released by OpenAI in 2019, is a decoder-only transformer "
            "trained on 40GB of internet text. The original transformer had "
            "6 encoder and 6 decoder layers. BERT-base has 12 attention heads "
            "and 110 million parameters."
        ),
        "qa": [
            ("What year was the transformer architecture introduced?", "2017"),
            ("What training objective does BERT use?", "Masked language modeling and next sentence prediction"),
            ("How many encoder layers does the original transformer have?", "6"),
            ("How many parameters does BERT-base have?", "110 million"),
            ("What data was GPT-2 trained on?", "40GB of internet text"),
        ],
    },
    {
        "topic": "space_exploration",
        "context": (
            "The Apollo 11 mission landed on the Moon on July 20, 1969. "
            "Neil Armstrong was the first human to walk on the Moon, followed "
            "by Buzz Aldrin. Michael Collins remained in the command module "
            "orbiting the Moon. The landing site was the Sea of Tranquility. "
            "The mission launched from Kennedy Space Center on July 16, 1969, "
            "and returned to Earth on July 24, 1969. The total mission duration "
            "was 8 days, 3 hours, and 18 minutes."
        ),
        "qa": [
            ("Who was the second human to walk on the Moon?", "Buzz Aldrin"),
            ("Where did Apollo 11 land on the Moon?", "Sea of Tranquility"),
            ("What did Michael Collins do during the mission?", "Remained in the command module orbiting the Moon"),
            ("When did Apollo 11 launch?", "July 16, 1969"),
            ("How long did the Apollo 11 mission last?", "8 days, 3 hours, and 18 minutes"),
        ],
    },
    {
        "topic": "climate_change",
        "context": (
            "The Paris Agreement was adopted in December 2015 and entered "
            "into force in November 2016. Its central aim is to limit global "
            "warming to well below 2 degrees Celsius above pre-industrial levels, "
            "with efforts to limit it to 1.5 degrees Celsius. The agreement "
            "requires countries to submit Nationally Determined Contributions (NDCs). "
            "As of 2023, 195 parties have signed the agreement. Carbon dioxide "
            "concentration in the atmosphere crossed 420 ppm in 2023, the highest "
            "level in at least 800,000 years."
        ),
        "qa": [
            ("What is the main temperature target of the Paris Agreement?", "Well below 2 degrees Celsius above pre-industrial levels"),
            ("When did the Paris Agreement enter into force?", "November 2016"),
            ("What are countries required to submit under the Paris Agreement?", "Nationally Determined Contributions (NDCs)"),
            ("What was the CO2 concentration in 2023?", "420 ppm"),
            ("How many parties have signed the Paris Agreement as of 2023?", "195"),
        ],
    },
    {
        "topic": "quantum_computing",
        "context": (
            "A qubit is the basic unit of quantum information, analogous to "
            "a classical bit. Unlike classical bits, qubits can exist in "
            "superpositions of 0 and 1. Quantum entanglement allows qubits to "
            "be correlated regardless of distance. Shor's algorithm, developed "
            "by Peter Shor in 1994, can factor large integers in polynomial time, "
            "threatening RSA encryption. Grover's algorithm provides a quadratic "
            "speedup for unstructured search problems. Google claimed quantum "
            "supremacy in 2019 with their 53-qubit Sycamore processor, completing "
            "a task in 200 seconds that would take a classical computer 10,000 years."
        ),
        "qa": [
            ("What is the quantum analogue of a classical bit?", "A qubit"),
            ("What encryption does Shor's algorithm threaten?", "RSA encryption"),
            ("When did Peter Shor develop Shor's algorithm?", "1994"),
            ("How many qubits did Google's Sycamore processor have?", "53"),
            ("What speedup does Grover's algorithm provide for search?", "Quadratic speedup"),
        ],
    },
    {
        "topic": "world_war_ii",
        "context": (
            "World War II began on September 1, 1939, when Germany invaded Poland. "
            "The United States entered the war after the Japanese attack on Pearl "
            "Harbor on December 7, 1941. The Battle of Stalingrad, lasting from "
            "August 1942 to February 1943, was the largest land battle in history "
            "and marked a turning point on the Eastern Front. D-Day, the Allied "
            "invasion of Normandy, took place on June 6, 1944. Germany surrendered "
            "on May 8, 1945 (V-E Day). Japan surrendered on September 2, 1945 "
            "(V-J Day), ending the war."
        ),
        "qa": [
            ("When did World War II begin?", "September 1, 1939"),
            ("What event caused the US to enter World War II?", "The Japanese attack on Pearl Harbor"),
            ("What was significant about the Battle of Stalingrad?", "It was the largest land battle in history and a turning point on the Eastern Front"),
            ("When did D-Day take place?", "June 6, 1944"),
            ("When did Germany surrender?", "May 8, 1945"),
        ],
    },
    {
        "topic": "biology",
        "context": (
            "DNA is a double-helix molecule composed of four nucleotide bases: "
            "adenine (A), thymine (T), guanine (G), and cytosine (C). "
            "A pairs with T, and G pairs with C. The human genome contains "
            "approximately 3 billion base pairs and about 20,000 protein-coding genes. "
            "The central dogma of molecular biology states that information flows "
            "from DNA to RNA to protein. Transcription converts DNA to mRNA in "
            "the nucleus, while translation converts mRNA to protein at ribosomes. "
            "The Human Genome Project was completed in 2003."
        ),
        "qa": [
            ("How many base pairs does the human genome contain?", "Approximately 3 billion"),
            ("Which base pairs with adenine in DNA?", "Thymine (T)"),
            ("Where does translation occur in the cell?", "At ribosomes"),
            ("When was the Human Genome Project completed?", "2003"),
            ("How many protein-coding genes does the human genome have?", "About 20,000"),
        ],
    },
    {
        "topic": "economics",
        "context": (
            "GDP (Gross Domestic Product) measures the total monetary value of "
            "all goods and services produced in a country within a specific period. "
            "Inflation refers to the general increase in prices over time, measured "
            "by the Consumer Price Index (CPI). The Federal Reserve, the central "
            "bank of the United States, was established in 1913. It uses tools "
            "such as the federal funds rate and open market operations to influence "
            "monetary policy. The Great Depression began in 1929 after the stock "
            "market crash and lasted until the late 1930s, with US unemployment "
            "reaching 25% at its peak."
        ),
        "qa": [
            ("What does GDP measure?", "The total monetary value of all goods and services produced in a country within a specific period"),
            ("What index measures inflation?", "Consumer Price Index (CPI)"),
            ("When was the Federal Reserve established?", "1913"),
            ("What was the peak US unemployment during the Great Depression?", "25%"),
            ("What tool does the Fed use to influence monetary policy?", "The federal funds rate and open market operations"),
        ],
    },
    {
        "topic": "computer_science",
        "context": (
            "The Turing machine, proposed by Alan Turing in 1936, is a mathematical "
            "model of computation. The P vs NP problem, one of the Millennium Prize "
            "Problems, asks whether every problem whose solution can be verified in "
            "polynomial time can also be solved in polynomial time. Dijkstra's "
            "algorithm finds the shortest path in a weighted graph and was published "
            "in 1959. The RSA encryption algorithm was invented by Rivest, Shamir, "
            "and Adleman in 1977. The first computer bug was an actual moth found "
            "in a Harvard Mark II relay in 1947 by Grace Hopper's team."
        ),
        "qa": [
            ("Who proposed the Turing machine and when?", "Alan Turing in 1936"),
            ("What does the P vs NP problem ask?", "Whether every problem whose solution can be verified in polynomial time can also be solved in polynomial time"),
            ("What does Dijkstra's algorithm compute?", "The shortest path in a weighted graph"),
            ("When was RSA encryption invented?", "1977"),
            ("What was the first computer bug?", "An actual moth found in a Harvard Mark II relay in 1947"),
        ],
    },
]


def load_factual_synthetic(
    cache_path: Optional[str] = None,
    max_samples: Optional[int] = 50,
    use_cache: bool = True,
) -> list[FactualSample]:
    """
    Load factual consistency tasks.

    Args:
        cache_path:  Where to save/load the JSON cache
        max_samples: Limit to N samples
        use_cache:   If True, load from cache if available

    Returns:
        List of FactualSample
    """
    resolved = cache_path or _CACHE_PATH

    if use_cache and os.path.exists(resolved):
        logger.info(f"Loading factual tasks from cache: {resolved}")
        return _load_cache(resolved, max_samples)

    samples = _build_samples(max_samples)
    _save_cache(samples, resolved)
    return samples


def _build_samples(max_samples: Optional[int]) -> list[FactualSample]:
    all_samples = []
    for i, fs in enumerate(_FACT_SETS):
        questions = [q for q, _ in fs["qa"]]
        answers = [a for _, a in fs["qa"]]
        all_samples.append(
            FactualSample(
                id=f"factual_{i:04d}_{fs['topic']}",
                context=fs["context"],
                questions=questions,
                answers=answers,
                topic=fs["topic"],
            )
        )
    if max_samples:
        all_samples = all_samples[:max_samples]
    logger.info(f"Built {len(all_samples)} factual consistency tasks")
    return all_samples


def _save_cache(samples: list[FactualSample], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [sample_to_dict(s) for s in samples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved factual tasks cache to {path}")


def _load_cache(path: str, max_samples: Optional[int]) -> list[FactualSample]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    samples = []
    for row in data:
        samples.append(
            FactualSample(
                id=row["id"],
                context=row["context"],
                questions=row["questions"],
                answers=row["answers"],
                topic=row["topic"],
            )
        )
    return samples


def sample_to_dict(s: FactualSample) -> dict:
    return {
        "id": s.id,
        "category": s.category,
        "topic": s.topic,
        "context": s.context,
        "questions": s.questions,
        "answers": s.answers,
    }
