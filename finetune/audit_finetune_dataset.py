"""
finetune/audit_finetune_dataset.py
Quick audit tool for fine-tuning dataset composition and mismatch risk.

Run:
    python finetune/audit_finetune_dataset.py
    python finetune/audit_finetune_dataset.py --dataset data_loaders/data/finetune_dataset.jsonl --warn-threshold 0.7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import yaml


def infer_category_from_text(text: str, explicit_category: str | None = None) -> str:
    if explicit_category:
        return explicit_category
    t = text.lower()
    if "<|user|>\nproblem:" in t:
        return "multistep_arithmetic"
    if "<|user|>\nread the following information carefully" in t:
        return "factual_consistency"
    if "<|user|>\ngoal:" in t:
        return "tool_use_planning"
    if "<|user|>\nconsider this hypothetical premise" in t:
        return "causal_counterfactual"
    if "physics" in t and "<|user|>\nproblem:" in t:
        return "physics_reasoning"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fine-tuning dataset category mix")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", default=None, help="Override dataset JSONL path")
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.70,
        help="Warn when one category exceeds this fraction",
    )
    args = parser.parse_args()

    with open(_ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_path = args.dataset or cfg["finetune"]["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = str(_ROOT / dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    counts = {
        "multistep_arithmetic": 0,
        "factual_consistency": 0,
        "tool_use_planning": 0,
        "causal_counterfactual": 0,
        "physics_reasoning": 0,
        "unknown": 0,
    }

    total = 0
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            cat = infer_category_from_text(
                row.get("text", ""),
                explicit_category=row.get("category"),
            )
            counts[cat] = counts.get(cat, 0) + 1

    print("=" * 64)
    print("FINE-TUNE DATASET AUDIT")
    print("=" * 64)
    print(f"dataset: {dataset_path}")
    print(f"total_examples: {total}")

    if total == 0:
        print("No examples found.")
        return

    print("\nCategory mix:")
    dominant_cat = None
    dominant_ratio = 0.0
    for cat in [
        "multistep_arithmetic",
        "factual_consistency",
        "tool_use_planning",
        "causal_counterfactual",
        "physics_reasoning",
        "unknown",
    ]:
        n = counts.get(cat, 0)
        ratio = n / total
        print(f"  - {cat}: {n} ({ratio:.1%})")
        if ratio > dominant_ratio:
            dominant_ratio = ratio
            dominant_cat = cat

    print("\nRisk checks:")
    if dominant_ratio >= args.warn_threshold and dominant_cat != "unknown":
        print(
            f"  WARNING: dominant category {dominant_cat} = {dominant_ratio:.1%}. "
            "This can cause cross-category regression."
        )
    else:
        print("  OK: no single-category dominance above threshold.")

    if counts.get("unknown", 0) > 0:
        print("  WARNING: unknown-format examples detected. Check prompt templates.")

    print("=" * 64)


if __name__ == "__main__":
    main()
