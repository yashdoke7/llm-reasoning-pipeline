"""
datasets/toolbench_loader.py
Synthetic tool-use and plan execution dataset.

No download required — all tasks are generated programmatically.
These test whether LLMs can produce logically valid, executable step-by-step
plans given a goal and a fixed toolset.

Optionally, if you place a JSON file at datasets/data/toolbench_subset.json,
that will be used instead. Format:
  [{"id": "...", "goal": "...", "tools": [...], "correct_steps": [...]}]
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_PATH = os.path.join(_HERE, "data", "toolbench_subset.json")


@dataclass
class ToolSample:
    id: str
    goal: str
    available_tools: list[dict]       # [{"name": str, "description": str, "params": [str]}]
    correct_plan: list[str]           # reference step sequence (for evaluation)
    constraints: list[str] = field(default_factory=list)
    category: str = "tool_use_planning"


# ── Built-in synthetic tasks ──────────────────────────────────

_TOOL_REGISTRY = {
    "search_web": {"description": "Search the web for information.", "params": ["query: str"]},
    "read_file": {"description": "Read content from a local file.", "params": ["filepath: str"]},
    "write_file": {"description": "Write content to a file.", "params": ["filepath: str", "content: str"]},
    "send_email": {"description": "Send an email.", "params": ["to: str", "subject: str", "body: str"]},
    "run_python": {"description": "Execute a Python code snippet.", "params": ["code: str"]},
    "query_database": {"description": "Run a SQL query on a database.", "params": ["query: str"]},
    "download_url": {"description": "Download content from a URL.", "params": ["url: str", "save_as: str"]},
    "summarize_text": {"description": "Summarize a block of text.", "params": ["text: str", "max_words: int"]},
    "translate_text": {"description": "Translate text to a target language.", "params": ["text: str", "target_lang: str"]},
    "compute_stats": {"description": "Compute descriptive statistics on a list of numbers.", "params": ["data: list[float]"]},
    "parse_pdf": {"description": "Extract text from a PDF file.", "params": ["filepath: str"]},
    "create_chart": {"description": "Create a chart from data and save as an image.", "params": ["data: dict", "chart_type: str", "save_as: str"]},
    "schedule_task": {"description": "Schedule a task to run at a given time.", "params": ["task_name: str", "run_at: str"]},
    "api_call": {"description": "Make an HTTP request to an external API.", "params": ["url: str", "method: str", "payload: dict"]},
}


_SYNTHETIC_TASKS = [
    {
        "goal": "Find the current weather in Mumbai, summarize it, and write the summary to a file called weather_report.txt.",
        "tools": ["search_web", "summarize_text", "write_file"],
        "correct_plan": [
            "Call search_web with query='current weather Mumbai' to retrieve weather data.",
            "Call summarize_text with the retrieved weather information to create a concise summary.",
            "Call write_file with filepath='weather_report.txt' and the summary as content.",
        ],
        "constraints": ["Must not send any emails.", "File must be named exactly weather_report.txt."],
    },
    {
        "goal": "Download the CSV from https://data.example.com/sales.csv, compute descriptive statistics on the 'revenue' column, and create a bar chart saved as revenue_chart.png.",
        "tools": ["download_url", "run_python", "compute_stats", "create_chart"],
        "correct_plan": [
            "Call download_url with url='https://data.example.com/sales.csv' and save_as='sales.csv'.",
            "Call run_python to read the CSV and extract the revenue column as a list of floats.",
            "Call compute_stats with the extracted revenue data.",
            "Call create_chart with the statistics data, chart_type='bar', and save_as='revenue_chart.png'.",
        ],
        "constraints": ["Do not query any database.", "Chart must be saved as a PNG."],
    },
    {
        "goal": "Read the report from report.pdf, translate it to Spanish, and email it to manager@company.com with subject 'Translated Report'.",
        "tools": ["parse_pdf", "translate_text", "send_email"],
        "correct_plan": [
            "Call parse_pdf with filepath='report.pdf' to extract the text content.",
            "Call translate_text with the extracted text and target_lang='Spanish'.",
            "Call send_email with to='manager@company.com', subject='Translated Report', and the translated text as body.",
        ],
        "constraints": ["Do not write any files to disk.", "Target language must be Spanish."],
    },
    {
        "goal": "Query the sales database for all transactions over $1000 in the last 30 days, then run a Python script to compute the total revenue and print it.",
        "tools": ["query_database", "run_python"],
        "correct_plan": [
            "Call query_database with query='SELECT * FROM transactions WHERE amount > 1000 AND date >= DATE_SUB(NOW(), INTERVAL 30 DAY)'.",
            "Call run_python with code that sums the 'amount' field from the query results and prints the total.",
        ],
        "constraints": ["Do not search the web.", "Must compute total using run_python, not query_database."],
    },
    {
        "goal": "Search for the latest research on transformer models, summarize the top 3 results, and schedule a task called 'read_papers' to run tomorrow at 9am.",
        "tools": ["search_web", "summarize_text", "schedule_task"],
        "correct_plan": [
            "Call search_web with query='latest transformer model research 2024'.",
            "Call summarize_text on the search results, limiting to the top 3 sources.",
            "Call schedule_task with task_name='read_papers' and run_at='tomorrow 09:00'.",
        ],
        "constraints": ["Must summarize exactly 3 results.", "Task must be scheduled for 9am."],
    },
    {
        "goal": "Make a POST request to https://api.service.com/predict with payload {model: 'v2', input: 'hello'}, then write the API response to output.txt.",
        "tools": ["api_call", "write_file"],
        "correct_plan": [
            "Call api_call with url='https://api.service.com/predict', method='POST', and payload={'model': 'v2', 'input': 'hello'}.",
            "Call write_file with filepath='output.txt' and the API response as content.",
        ],
        "constraints": ["Must use POST method.", "Response must be saved to output.txt."],
    },
    {
        "goal": "Read a Python source file called data_processor.py, run it to verify it executes without errors, then send the execution result by email to dev@team.com.",
        "tools": ["read_file", "run_python", "send_email"],
        "correct_plan": [
            "Call read_file with filepath='data_processor.py' to get the source code.",
            "Call run_python with the source code and capture stdout/stderr.",
            "Call send_email with to='dev@team.com', subject='Execution Result', and the captured output as body.",
        ],
        "constraints": ["Email recipient must be dev@team.com.", "Do not modify the file."],
    },
    {
        "goal": "Search for stock price of AAPL, compute statistics on its last 30 prices (search results), and create a line chart saved as aapl_chart.png.",
        "tools": ["search_web", "compute_stats", "create_chart"],
        "correct_plan": [
            "Call search_web with query='AAPL stock price history last 30 days'.",
            "Extract numerical price values from the search results.",
            "Call compute_stats with the extracted prices.",
            "Call create_chart with the price data, chart_type='line', and save_as='aapl_chart.png'.",
        ],
        "constraints": ["Chart must be a line chart.", "Must use at least 10 data points."],
    },
]


def load_toolbench(
    path: Optional[str] = None,
    max_samples: Optional[int] = 50,
) -> list[ToolSample]:
    """
    Load tool-use planning samples.

    Priority:
      1. Generated JSONL (from data_loaders/generate_eval_datasets.py) — best option
      2. Custom JSON file if it exists at the configured path
      3. Built-in synthetic tasks (8 hardcoded) — last resort

    Args:
        path:        Path to dataset file (.jsonl or .json)
        max_samples: Limit to N samples

    Returns:
        List of ToolSample
    """
    resolved = path or _CUSTOM_PATH

    if os.path.exists(resolved):
        logger.info(f"Loading tool-use tasks from {resolved}")
        return _load_from_file(resolved, max_samples)

    # Check for generated JSONL in default location
    generated_path = os.path.join(_HERE, "data", "generated_tooluse.jsonl")
    if os.path.exists(generated_path):
        logger.info(f"Loading generated tool-use tasks from {generated_path}")
        return _load_from_file(generated_path, max_samples)

    logger.warning(
        "No tool-use dataset found. Using built-in synthetic tasks (8 samples only). "
        "Generate more: python data_loaders/generate_eval_datasets.py --categories tool_use_planning"
    )
    return _build_synthetic(max_samples)


def _load_from_file(path: str, max_samples: Optional[int]) -> list[ToolSample]:
    samples = []
    # JSONL format (generated)
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and len(samples) >= max_samples:
                    break
                try:
                    row = json.loads(line.strip())
                    tools = row.get("available_tools", [])
                    if not tools:
                        tools = [
                            _TOOL_REGISTRY.get(t, {"name": t, "description": t, "params": []})
                            for t in row.get("tools", [])
                        ]
                    samples.append(ToolSample(
                        id=row.get("id", f"tool_gen_{i:04d}"),
                        goal=row["goal"],
                        available_tools=tools,
                        correct_plan=row.get("correct_plan", row.get("correct_steps", [])),
                        constraints=row.get("constraints", []),
                    ))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skipping malformed line {i}: {e}")
        logger.info(f"Loaded {len(samples)} tool-use tasks from JSONL")
        return samples

    # JSON array format
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    for i, row in enumerate(data):
        tools = [_TOOL_REGISTRY.get(t, {"name": t, "description": t, "params": []}) for t in row.get("tools", [])]
        samples.append(
            ToolSample(
                id=row.get("id", f"tool_{i:04d}"),
                goal=row["goal"],
                available_tools=tools,
                correct_plan=row.get("correct_steps", row.get("correct_plan", [])),
                constraints=row.get("constraints", []),
            )
        )
    return samples


def _build_synthetic(max_samples: Optional[int]) -> list[ToolSample]:
    data = _SYNTHETIC_TASKS
    if max_samples:
        data = data[:max_samples]
    samples = []
    for i, row in enumerate(data):
        tools = [
            {"name": t, **_TOOL_REGISTRY.get(t, {"description": t, "params": []})}
            for t in row["tools"]
        ]
        samples.append(
            ToolSample(
                id=f"tool_synthetic_{i:04d}",
                goal=row["goal"],
                available_tools=tools,
                correct_plan=row["correct_plan"],
                constraints=row.get("constraints", []),
            )
        )
    logger.info(f"Built {len(samples)} synthetic tool-use tasks")
    return samples


def format_tools_for_prompt(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        params = ", ".join(t.get("params", []))
        lines.append(f"  - {t['name']}({params}): {t.get('description', '')}")
    return "\n".join(lines)


def sample_to_dict(s: ToolSample) -> dict:
    return {
        "id": s.id,
        "category": s.category,
        "goal": s.goal,
        "available_tools": s.available_tools,
        "correct_plan": s.correct_plan,
        "constraints": s.constraints,
    }
