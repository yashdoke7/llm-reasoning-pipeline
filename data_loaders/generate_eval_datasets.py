"""
data_loaders/generate_eval_datasets.py
Generates diverse evaluation datasets for the 3 non-arithmetic categories
using a configurable LLM (GPT-4.1, Ollama qwen2.5:7b, etc.)

Why this exists:
  The hardcoded synthetic datasets (8 factual, 8 tool-use, 10 counterfactual)
  are too small and static to be a real benchmark. This script generates
  50-100 diverse examples per category using a strong model, saves them to
  JSONL, and makes the evaluation credible.

Run:
    # Using Ollama (free, local):
    python data_loaders/generate_eval_datasets.py --provider ollama --model qwen2.5:7b

    # Using OpenAI (GPT-4.1, best quality):
    python data_loaders/generate_eval_datasets.py --provider openai --model gpt-4.1

    # Generate only one category:
    python data_loaders/generate_eval_datasets.py --categories causal_counterfactual --samples 50

Output:
    data_loaders/data/generated_factual.jsonl
    data_loaders/data/generated_tooluse.jsonl
    data_loaders/data/generated_counterfactual.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_DATA_DIR = _HERE / "data"

# ─────────────────────────────────────────────────────────────────
# Generation prompts — instruct the LLM to produce structured JSON
# ─────────────────────────────────────────────────────────────────

_FACTUAL_GEN_SYSTEM = """You generate factual consistency evaluation tasks.

Each task consists of:
- A context paragraph (4-8 sentences) containing specific, verifiable facts
- 3-5 questions whose answers are EXPLICITLY stated in the context
- The correct answer for each question (short, extractable from the context)

Rules:
- Vary the domain each time (science, history, geography, technology, biology, sports, economics, etc.)
- Each answer must be a short phrase directly derivable from the context
- Do NOT ask questions whose answers require outside knowledge — only from the context
- Make questions specific (dates, numbers, names, procedures)

Respond ONLY with valid JSON:
{
  "topic": "<domain topic>",
  "context": "<paragraph with facts>",
  "qa": [
    {"question": "<question>", "answer": "<short answer from context>"},
    ...
  ]
}"""

_TOOLUSE_GEN_SYSTEM = """You generate tool-use planning evaluation tasks.

Each task consists of:
- A goal that requires 2-4 sequential tool calls
- The exact list of available tools needed (from the provided registry)
- Constraints that must be respected
- The correct step sequence (exact tool calls with parameters)

Available tools (use only these):
search_web(query), read_file(filepath), write_file(filepath, content),
send_email(to, subject, body), run_python(code), query_database(query),
download_url(url, save_as), summarize_text(text, max_words),
translate_text(text, target_lang), compute_stats(data),
parse_pdf(filepath), create_chart(data, chart_type, save_as),
schedule_task(task_name, run_at), api_call(url, method, payload)

Rules:
- Goal must require 2-4 tool calls in a logical sequence
- Each step must use exactly one tool with explicit parameter values
- Constraints must be specific and checkable
- Vary the domain (data analysis, communication, file management, research, etc.)

Respond ONLY with valid JSON:
{
  "goal": "<clear goal description>",
  "tools": ["tool1", "tool2", ...],
  "constraints": ["constraint1", "constraint2"],
  "correct_plan": [
    "Call tool1 with param1='value1' to <purpose>.",
    "Call tool2 with param1='...' to <purpose>.",
    ...
  ]
}"""

_COUNTERFACTUAL_GEN_SYSTEM = """You generate causal counterfactual reasoning evaluation tasks.

Each task has:
- A counterfactual premise ("If X had not happened / did not exist...")
- A question about consequences of that premise
- The correct answer (what would logically follow)
- 3 distractors (plausible but wrong answers)

Rules:
- Vary domains: science, history, technology, biology, society, economics
- The correct answer must follow logically from the premise
- Distractors must be plausible but clearly wrong given careful reasoning
- Avoid trivial premises — make them require actual causal reasoning
- Keep answers to 1-2 sentences

Respond ONLY with valid JSON:
{
  "premise": "If <counterfactual condition>...",
  "question": "<question about consequences>",
  "correct_answer": "<1-2 sentence correct answer>",
  "distractors": ["<wrong1>", "<wrong2>", "<wrong3>"]
}"""

# ─────────────────────────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────────────────────────

def _generate_factual_task(client, model: str) -> dict | None:
    """Generate one factual consistency task."""
    try:
        domains = [
            "astrophysics", "medieval history", "marine biology", "computer architecture",
            "Renaissance art", "organic chemistry", "ancient civilizations", "modern economics",
            "neuroscience", "climate science", "electrical engineering", "linguistics",
            "evolutionary biology", "political philosophy", "materials science",
            "world geography", "music theory", "genomics", "thermodynamics", "archaeology",
        ]
        import random
        domain = random.choice(domains)

        raw = client.complete_json(
            model=model,
            user_prompt=f"Generate a factual consistency task about: {domain}",
            system_prompt=_FACTUAL_GEN_SYSTEM,
            temperature=0.9,
            max_tokens=800,
        )

        # Validate structure
        if not all(k in raw for k in ["topic", "context", "qa"]):
            return None
        if not raw["qa"] or len(raw["qa"]) < 2:
            return None

        return {
            "category": "factual_consistency",
            "topic": raw["topic"],
            "context": raw["context"],
            "questions": [item["question"] for item in raw["qa"]],
            "answers": [item["answer"] for item in raw["qa"]],
        }
    except Exception as e:
        logger.debug(f"Factual generation error: {e}")
        return None


def _generate_tooluse_task(client, model: str) -> dict | None:
    """Generate one tool-use planning task."""
    try:
        raw = client.complete_json(
            model=model,
            user_prompt="Generate a new tool-use planning task. Be creative with the goal and vary the domain.",
            system_prompt=_TOOLUSE_GEN_SYSTEM,
            temperature=0.9,
            max_tokens=600,
        )

        if not all(k in raw for k in ["goal", "tools", "correct_plan", "constraints"]):
            return None
        if len(raw["correct_plan"]) < 2:
            return None

        # Build available_tools list format matching ToolSample
        _TOOL_DESCRIPTIONS = {
            "search_web": {"description": "Search the web for information.", "params": ["query: str"]},
            "read_file": {"description": "Read content from a local file.", "params": ["filepath: str"]},
            "write_file": {"description": "Write content to a file.", "params": ["filepath: str", "content: str"]},
            "send_email": {"description": "Send an email.", "params": ["to: str", "subject: str", "body: str"]},
            "run_python": {"description": "Execute a Python code snippet.", "params": ["code: str"]},
            "query_database": {"description": "Run a SQL query.", "params": ["query: str"]},
            "download_url": {"description": "Download content from a URL.", "params": ["url: str", "save_as: str"]},
            "summarize_text": {"description": "Summarize a block of text.", "params": ["text: str", "max_words: int"]},
            "translate_text": {"description": "Translate text.", "params": ["text: str", "target_lang: str"]},
            "compute_stats": {"description": "Compute descriptive statistics.", "params": ["data: list[float]"]},
            "parse_pdf": {"description": "Extract text from a PDF.", "params": ["filepath: str"]},
            "create_chart": {"description": "Create a chart image.", "params": ["data: dict", "chart_type: str", "save_as: str"]},
            "schedule_task": {"description": "Schedule a task.", "params": ["task_name: str", "run_at: str"]},
            "api_call": {"description": "Make an HTTP request.", "params": ["url: str", "method: str", "payload: dict"]},
        }

        available_tools = []
        for t in raw["tools"]:
            if t in _TOOL_DESCRIPTIONS:
                available_tools.append({"name": t, **_TOOL_DESCRIPTIONS[t]})

        return {
            "category": "tool_use_planning",
            "goal": raw["goal"],
            "available_tools": available_tools,
            "constraints": raw["constraints"],
            "correct_plan": raw["correct_plan"],
        }
    except Exception as e:
        logger.debug(f"Tool-use generation error: {e}")
        return None


def _generate_counterfactual_task(client, model: str) -> dict | None:
    """Generate one counterfactual reasoning task."""
    try:
        raw = client.complete_json(
            model=model,
            user_prompt="Generate a new causal counterfactual task. Vary the domain and difficulty.",
            system_prompt=_COUNTERFACTUAL_GEN_SYSTEM,
            temperature=0.9,
            max_tokens=500,
        )

        if not all(k in raw for k in ["premise", "question", "correct_answer", "distractors"]):
            return None

        return {
            "category": "causal_counterfactual",
            "premise": raw["premise"],
            "question": raw["question"],
            "correct_answer": raw["correct_answer"],
            "distractors": raw.get("distractors", []),
        }
    except Exception as e:
        logger.debug(f"Counterfactual generation error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────

_GENERATORS = {
    "factual_consistency": (_generate_factual_task, "generated_factual.jsonl"),
    "tool_use_planning": (_generate_tooluse_task, "generated_tooluse.jsonl"),
    "causal_counterfactual": (_generate_counterfactual_task, "generated_counterfactual.jsonl"),
}


def generate_category(
    category: str,
    client,
    model: str,
    target: int = 50,
    output_path: str = None,
) -> int:
    """
    Generate `target` examples for a category and save to JSONL.
    Returns number of examples successfully generated.
    """
    gen_fn, default_filename = _GENERATORS[category]
    out = output_path or str(_DATA_DIR / default_filename)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    written = 0
    attempts = 0
    max_attempts = target * 3  # allow retries for failures

    logger.info(f"Generating {target} examples for '{category}' → {out}")

    with open(out, "w", encoding="utf-8") as f:
        with tqdm(total=target, desc=category) as pbar:
            while written < target and attempts < max_attempts:
                attempts += 1
                result = gen_fn(client, model)
                if result:
                    result["id"] = f"{category}_gen_{written:04d}"
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    written += 1
                    pbar.update(1)
                else:
                    time.sleep(0.5)

    logger.info(f"  {written}/{target} generated ({attempts} attempts)")
    return written


def main():
    parser = argparse.ArgumentParser(description="Generate diverse eval datasets using an LLM")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--provider", choices=["groq", "ollama", "openai"], default="ollama",
        help="Provider to use for generation. openai=GPT-4.1, ollama=local (qwen2.5:7b recommended)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Model ID. Defaults: ollama=qwen2.5:7b, openai=gpt-4.1, groq=llama-3.3-70b-versatile"
    )
    parser.add_argument(
        "--categories", nargs="+",
        default=["factual_consistency", "tool_use_planning", "causal_counterfactual"],
        choices=["factual_consistency", "tool_use_planning", "causal_counterfactual"],
        help="Which categories to generate"
    )
    parser.add_argument("--samples", type=int, default=50, help="Examples per category")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config_path = _ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from models import get_client
    client = get_client(cfg, provider=args.provider)

    # Default models per provider
    _defaults = {
        "ollama": "qwen2.5:7b",
        "openai": "gpt-4.1",
        "groq": "llama-3.3-70b-versatile",
    }
    model = args.model or _defaults[args.provider]

    logger.info("=" * 60)
    logger.info("DATASET GENERATION")
    logger.info(f"  Provider: {args.provider} / Model: {model}")
    logger.info(f"  Categories: {args.categories}")
    logger.info(f"  Samples per category: {args.samples}")
    logger.info("=" * 60)

    total = 0
    for cat in args.categories:
        n = generate_category(cat, client, model, target=args.samples)
        total += n
        logger.info(f"  ✓ {cat}: {n} examples")

    logger.info(f"\nTotal generated: {total} examples")
    logger.info(f"Files saved to: {_DATA_DIR}")
    logger.info("\nNext step: update configs/config.yaml to point datasets at generated files")
    logger.info("  factual_synthetic.path: data_loaders/data/generated_factual.jsonl")
    logger.info("  toolbench.path:         data_loaders/data/generated_tooluse.jsonl")
    logger.info("  crass.path:             data_loaders/data/generated_counterfactual.jsonl")


if __name__ == "__main__":
    main()
