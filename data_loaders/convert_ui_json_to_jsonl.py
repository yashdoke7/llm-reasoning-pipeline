"""
data_loaders/convert_ui_json_to_jsonl.py
Convert JSON arrays generated in model UIs (ChatGPT/Gemini) into
pipeline-compatible JSONL files.
"""
from __future__ import annotations

import argparse
import json
import os


SCHEMA_HINTS = {
    "factual_consistency": ("context", "questions", "answers"),
    "tool_use_planning": ("goal", "available_tools", "correct_plan"),
    "causal_counterfactual": ("premise", "question", "correct_answer"),
    "physics_reasoning": ("question", "ground_truth"),
}


def _normalize_record(category: str, row: dict, idx: int) -> dict:
    out = dict(row)
    out.setdefault("id", f"{category}_gen_{idx:04d}")
    out["category"] = category

    if category == "physics_reasoning":
        question = (out.get("question") or out.get("problem") or out.get("prompt") or "").strip()
        answer = (out.get("ground_truth") or out.get("final_answer") or out.get("answer") or "").strip()
        out["question"] = question
        out["ground_truth"] = answer
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UI-generated JSON array to JSONL")
    parser.add_argument("--input", required=True, help="Input JSON array path")
    parser.add_argument(
        "--category",
        required=True,
        choices=["factual_consistency", "tool_use_planning", "causal_counterfactual", "physics_reasoning"],
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            record = _normalize_record(args.category, row, i)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    req = ", ".join(SCHEMA_HINTS[args.category])
    print(f"Wrote {written} rows to {args.output}")
    print(f"Required keys for {args.category}: {req}")


if __name__ == "__main__":
    main()

