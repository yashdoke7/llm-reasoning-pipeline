"""
experiments/generate_charts.py
Generate static charts from eval_results_*.json into outputs/charts.

Run:
    python experiments/generate_charts.py
    python experiments/generate_charts.py --latest-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_eval_files(outputs_dir: str, latest_only: bool) -> list[str]:
    files = [
        os.path.join(outputs_dir, f)
        for f in os.listdir(outputs_dir)
        if f.startswith("eval_results_") and f.endswith(".json")
    ]
    files = sorted(files)
    if latest_only and files:
        return [files[-1]]
    return files


def to_rows(eval_json: dict) -> tuple[list[dict], list[dict]]:
    per_run = eval_json.get("per_run_metrics", [])
    model_summary = eval_json.get("model_summary", [])

    cat_rows = []
    for r in per_run:
        cat_rows.append(
            {
                "model": r["model"],
                "category": r["category"],
                "accuracy": r["final_accuracy"],
                "step_failure": r["step_failure_rate"],
                "error_prop": r["error_propagation_rate"],
            }
        )

    model_rows = []
    for m in model_summary:
        model_rows.append(
            {
                "model": m["model"],
                "accuracy": m["overall_final_accuracy"],
                "step_failure": m["overall_step_failure_rate"],
                "error_prop": m["overall_error_propagation_rate"],
                "tasks": m["total_tasks"],
            }
        )

    return cat_rows, model_rows


def save_category_bars(df_cat: pd.DataFrame, out_png: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    pivot_acc = df_cat.pivot(index="category", columns="model", values="accuracy")
    pivot_acc.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Accuracy by Category")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)

    pivot_sf = df_cat.pivot(index="category", columns="model", values="step_failure")
    pivot_sf.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Step Failure by Category")
    axes[1].set_ylabel("Step Failure")

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_scatter(df_model: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for _, row in df_model.iterrows():
        ax.scatter(row["step_failure"], row["accuracy"], s=120)
        ax.annotate(row["model"], (row["step_failure"], row["accuracy"]), fontsize=9)

    ax.set_title("Accuracy vs Step Failure")
    ax.set_xlabel("Step Failure")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate charts from eval results")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--latest-only", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(str(_ROOT / args.config))
    outputs_dir = str(_ROOT / cfg["paths"]["outputs"])
    charts_dir = str(_ROOT / cfg["paths"]["charts"])
    os.makedirs(charts_dir, exist_ok=True)

    files = iter_eval_files(outputs_dir, args.latest_only)
    if not files:
        print("No eval_results_*.json files found.")
        return

    generated = 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        cat_rows, model_rows = to_rows(data)
        if not cat_rows or not model_rows:
            print(f"Skipping empty/invalid run: {os.path.basename(fp)}")
            continue

        run_tag = os.path.basename(fp).replace("eval_results_", "").replace(".json", "")
        df_cat = pd.DataFrame(cat_rows)
        df_model = pd.DataFrame(model_rows)

        out_cat = os.path.join(charts_dir, f"category_bars_{run_tag}.png")
        out_scatter = os.path.join(charts_dir, f"scatter_{run_tag}.png")

        save_category_bars(df_cat, out_cat)
        save_scatter(df_model, out_scatter)
        generated += 2
        print(f"Generated charts for {os.path.basename(fp)}")

    print(f"Done. Generated {generated} chart files in {charts_dir}")


if __name__ == "__main__":
    main()
