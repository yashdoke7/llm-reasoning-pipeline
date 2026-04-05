from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_eval(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def model_summary_df(data: dict) -> pd.DataFrame:
    rows = []
    for m in data.get("model_summary", []):
        rows.append(
            {
                "model": m["model"],
                "accuracy": m["overall_final_accuracy"],
                "step_failure": m["overall_step_failure_rate"],
                "error_prop": m["overall_error_propagation_rate"],
            }
        )
    return pd.DataFrame(rows)


def category_df(data: dict) -> pd.DataFrame:
    rows = []
    for r in data.get("per_run_metrics", []):
        rows.append(
            {
                "model": r["model"],
                "category": r["category"],
                "accuracy": r["final_accuracy"],
                "step_failure": r["step_failure_rate"],
                "error_prop": r["error_propagation_rate"],
            }
        )
    return pd.DataFrame(rows)


def plot_overall(df: pd.DataFrame, out: Path) -> None:
    # Convert all metrics to higher-is-better scale for clean visual comparison.
    plot_df = pd.DataFrame(
        {
            "model": df["model"],
            "accuracy": df["accuracy"],
            "step_quality": 1.0 - df["step_failure"],
            "stability": 1.0 - df["error_prop"],
        }
    ).melt(id_vars="model", var_name="metric", value_name="value")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.barplot(data=plot_df, x="metric", y="value", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Overall Performance (Higher Is Better)")
    ax.legend(title="Model")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_category_accuracy_delta(df_cat: pd.DataFrame, out: Path) -> None:
    pivot = df_cat.pivot(index="category", columns="model", values="accuracy")
    if pivot.shape[1] < 2:
        return
    base = pivot.columns[0]
    ft = pivot.columns[1]
    delta = (pivot[ft] - pivot[base]).sort_values(ascending=False)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in delta.values]
    ax.bar(delta.index, delta.values, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Accuracy Delta (FT - Base)")
    ax.set_xlabel("")
    ax.set_title("Category-wise Fine-tuning Gain")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_category_heatmap(df_cat: pd.DataFrame, out: Path) -> None:
    heat = df_cat.pivot_table(index="category", columns="model", values="accuracy")
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title("Accuracy Heatmap by Category")
    ax.set_xlabel("Model")
    ax.set_ylabel("Category")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-style charts from one eval json")
    parser.add_argument("--input", default="outputs/final_fixed20_groq8b.json")
    parser.add_argument("--outdir", default="outputs/charts/paper")
    args = parser.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_eval(inp)
    df_m = model_summary_df(data)
    df_c = category_df(data)

    plot_overall(df_m, outdir / "overall_metrics.png")
    plot_category_accuracy_delta(df_c, outdir / "category_accuracy_delta.png")
    plot_category_heatmap(df_c, outdir / "category_accuracy_heatmap.png")

    print("Generated:")
    print(outdir / "overall_metrics.png")
    print(outdir / "category_accuracy_delta.png")
    print(outdir / "category_accuracy_heatmap.png")


if __name__ == "__main__":
    main()
