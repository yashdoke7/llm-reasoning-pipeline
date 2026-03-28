"""
finetune/prepare_physics_dataset.py
Convert curated physics JSON into:
  1) training JSONL in SFT format
  2) evaluation JSONL for physics_reasoning benchmark

Input JSON expected: list of objects with keys like:
  - problem OR question (required)
  - reasoning_trace OR solution (required for train split)
  - final_answer OR ground_truth OR answer (recommended for eval)
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


SYSTEM_PROMPT = """You are an expert physics tutor. Generate a perfect, step-by-step reasoning trace for the given problem.

Requirements:
- Number each step: Step 1:, Step 2:, etc.
- Each step must contain exactly one logical operation or inference
- Each step must be explicit, not skipping any computation
- End with: Final Answer: <answer>
- The final answer MUST be correct
- Do not make any factual errors"""


def _extract_problem(row: dict) -> str:
    return (row.get("problem") or row.get("question") or "").strip()


def _extract_trace(row: dict) -> str:
    return (row.get("reasoning_trace") or row.get("solution") or "").strip()


def _extract_answer(row: dict) -> str:
    return (
        row.get("ground_truth")
        or row.get("final_answer")
        or row.get("answer")
        or ""
    ).strip()


def _to_train_entry(problem: str, trace: str, idx: int) -> dict:
    user = f"Problem: {problem}"
    text = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user}\n"
        f"<|assistant|>\n{trace.strip()}"
    )
    return {
        "id": f"physics_train_{idx:04d}",
        "category": "physics_reasoning",
        "source": "physics_curated",
        "text": text,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": trace.strip()},
        ],
    }


def _to_eval_entry(problem: str, answer: str, idx: int) -> dict:
    return {
        "id": f"physics_eval_{idx:04d}",
        "category": "physics_reasoning",
        "question": problem,
        "ground_truth": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare physics train/eval datasets")
    parser.add_argument("--input", default="physics_problems.json", help="Input JSON path")
    parser.add_argument("--train-output", default="data_loaders/data/ft_physics.jsonl")
    parser.add_argument("--eval-output", default="data_loaders/data/physics_eval.jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of problems")

    rows = []
    for row in data:
        if not isinstance(row, dict):
            continue
        problem = _extract_problem(row)
        trace = _extract_trace(row)
        answer = _extract_answer(row)
        if not problem:
            continue
        rows.append({"problem": problem, "trace": trace, "answer": answer})

    if len(rows) < 10:
        raise ValueError("Need at least 10 valid rows to make train/eval splits")

    random.seed(args.seed)
    random.shuffle(rows)
    split = max(1, min(len(rows) - 1, int(len(rows) * args.train_ratio)))
    train_rows = rows[:split]
    eval_rows = rows[split:]

    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.eval_output), exist_ok=True)

    train_written = 0
    with open(args.train_output, "w", encoding="utf-8") as f:
        for i, row in enumerate(train_rows):
            if not row["trace"]:
                continue
            entry = _to_train_entry(row["problem"], row["trace"], i)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            train_written += 1

    eval_written = 0
    with open(args.eval_output, "w", encoding="utf-8") as f:
        for i, row in enumerate(eval_rows):
            if not row["answer"]:
                continue
            entry = _to_eval_entry(row["problem"], row["answer"], i)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            eval_written += 1

    print(f"Prepared physics datasets from {input_path}")
    print(f"  Train JSONL: {args.train_output} ({train_written} rows)")
    print(f"  Eval  JSONL: {args.eval_output} ({eval_written} rows)")


if __name__ == "__main__":
    main()

