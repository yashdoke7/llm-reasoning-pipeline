from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_example(row: dict, source_name: str) -> dict | None:
    msgs = row.get("messages") or []
    if not isinstance(msgs, list) or not msgs:
        return None

    user_text = ""
    assistant_text = ""
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if role == "user" and content:
            user_text = content
        if role == "assistant" and content:
            assistant_text = content

    if not user_text or not assistant_text:
        return None

    category = row.get("category", "unknown")
    if category == "unknown":
        return None

    ex_id = row.get("id")
    if not ex_id:
        ex_id = f"{source_name}_{abs(hash((category, user_text[:120], assistant_text[:120]))) % 10**12}"

    return {
        "id": str(ex_id),
        "category": str(category),
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "source": source_name,
    }


def dedupe_examples(rows: list[dict]) -> list[dict]:
    seen = set()
    out: list[dict] = []
    for r in rows:
        msgs = r.get("messages") or []
        if len(msgs) < 2:
            continue
        key = (
            r.get("category", "unknown"),
            msgs[0].get("content", "").strip(),
            msgs[-1].get("content", "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def sample_by_quota(by_cat: dict[str, list[dict]], quotas: dict[str, int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected: list[dict] = []

    for cat, need in quotas.items():
        pool = list(by_cat.get(cat, []))
        if not pool:
            continue
        rng.shuffle(pool)
        selected.extend(pool[: min(len(pool), need)])

    rng.shuffle(selected)
    return selected


def _resolve_default_source_files() -> list[Path]:
    candidates = [
        ROOT / "finetune/data/archive_legacy_v5/generated_from_failures_fixed.jsonl",
        ROOT / "finetune/data/archive_legacy_v5/targeted_dataset_v4_run.jsonl",
        ROOT / "finetune/data/archive_legacy_v5/baseline_weighted_v5.jsonl",
        ROOT / "finetune/data/generated_from_failures_fixed.jsonl",
        ROOT / "finetune/data/targeted_dataset_v4_run.jsonl",
        ROOT / "finetune/data/baseline_weighted_v5.jsonl",
    ]
    return [p for p in candidates if p.exists()]


def _extract_baseline_metrics(
    baseline_results_path: Path,
    baseline_model: str | None,
) -> tuple[str, dict[str, dict[str, float]]]:
    raw = json.loads(baseline_results_path.read_text(encoding="utf-8"))
    model_summary = raw.get("model_summary", []) or []
    failure_report = raw.get("failure_report", {}) or {}

    chosen_model = baseline_model or failure_report.get("finetune_target_model")
    if not chosen_model and model_summary:
        chosen_model = model_summary[0].get("model")
    if not chosen_model:
        raise ValueError("Unable to determine baseline model from baseline results.")

    target = None
    for row in model_summary:
        if row.get("model") == chosen_model:
            target = row
            break
    if target is None:
        available = [r.get("model") for r in model_summary]
        raise ValueError(
            f"Baseline model '{chosen_model}' not found in baseline results. "
            f"Available models: {available}"
        )

    out: dict[str, dict[str, float]] = {}
    for cat, vals in (target.get("per_category", {}) or {}).items():
        out[cat] = {
            "step_failure_rate": float(vals.get("step_failure_rate", 0.0)),
            "final_accuracy": float(vals.get("final_accuracy", 0.0)),
        }

    return chosen_model, out


def _compute_auto_quotas(
    pool_by_category: dict[str, list[dict]],
    baseline_metrics_path: Path,
    baseline_model: str | None,
    target_total: int | None,
    min_total: int,
    max_total: int,
    usage_ratio: float,
    min_per_category: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    pool_counts = {k: len(v) for k, v in pool_by_category.items()}
    baseline_model_used, baseline_metrics = _extract_baseline_metrics(
        baseline_results_path=baseline_metrics_path,
        baseline_model=baseline_model,
    )

    categories = [c for c in baseline_metrics.keys() if pool_counts.get(c, 0) > 0]
    if not categories:
        raise ValueError("No overlapping categories between baseline metrics and available dataset pool.")

    # Weakness score: larger means category deserves more fine-tune data.
    raw_scores: dict[str, float] = {}
    for cat in categories:
        sf = baseline_metrics[cat]["step_failure_rate"]
        acc = baseline_metrics[cat]["final_accuracy"]
        raw_scores[cat] = 0.65 * sf + 0.35 * (1.0 - acc)

    score_sum = sum(raw_scores.values()) or 1.0
    weights = {cat: raw_scores[cat] / score_sum for cat in categories}

    available_total = sum(pool_counts.get(cat, 0) for cat in categories)
    if target_total is None:
        proposed = int(round(available_total * usage_ratio))
        total = max(min_total, min(max_total, proposed))
        total = min(total, available_total)
    else:
        total = max(min_total, min(max_total, target_total))
        total = min(total, available_total)

    quotas = {cat: min(pool_counts[cat], min_per_category) for cat in categories}
    assigned = sum(quotas.values())

    # If minimum floor exceeds total budget, shrink floors proportionally.
    if assigned > total:
        quotas = {cat: 0 for cat in categories}
        assigned = 0

    remaining = total - assigned
    if remaining > 0:
        # First pass by weakness weights.
        for cat in categories:
            extra = int(remaining * weights[cat])
            cap = pool_counts[cat] - quotas[cat]
            take = min(cap, max(0, extra))
            quotas[cat] += take

    # Fill leftovers using highest-weight categories with spare capacity.
    assigned = sum(quotas.values())
    leftover = max(0, total - assigned)
    if leftover > 0:
        order = sorted(categories, key=lambda c: weights[c], reverse=True)
        i = 0
        while leftover > 0 and order:
            cat = order[i % len(order)]
            if quotas[cat] < pool_counts[cat]:
                quotas[cat] += 1
                leftover -= 1
            i += 1
            if i > total * 10:
                break

    report = {
        "baseline_model": baseline_model_used,
        "baseline_metrics_path": str(baseline_metrics_path.relative_to(ROOT)),
        "pool_counts": pool_counts,
        "raw_weakness_scores": raw_scores,
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "available_total": available_total,
        "target_total": total,
        "min_per_category": min_per_category,
        "usage_ratio": usage_ratio,
    }
    return quotas, report


def build_finetune_dataset(
    seed: int,
    source_files: list[Path],
    baseline_metrics_path: Path,
    baseline_model: str | None,
    target_total: int | None,
    min_total: int,
    max_total: int,
    usage_ratio: float,
    min_per_category: int,
) -> tuple[list[dict], dict]:

    merged: list[dict] = []
    source_stats: dict[str, int] = {}

    for p in source_files:
        rows = load_jsonl(p)
        source_stats[str(p.relative_to(ROOT))] = len(rows)
        for r in rows:
            norm = normalize_example(r, source_name=p.stem)
            if norm:
                merged.append(norm)

    merged = dedupe_examples(merged)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in merged:
        by_cat[r["category"]].append(r)

    quotas, auto_report = _compute_auto_quotas(
        pool_by_category=by_cat,
        baseline_metrics_path=baseline_metrics_path,
        baseline_model=baseline_model,
        target_total=target_total,
        min_total=min_total,
        max_total=max_total,
        usage_ratio=usage_ratio,
        min_per_category=min_per_category,
    )

    curated = sample_by_quota(by_cat, quotas, seed=seed)

    quality = {
        "final_answer_like": 0,
        "step_format_like": 0,
        "total": len(curated),
    }
    for r in curated:
        asst = r["messages"][-1]["content"]
        low = asst.lower()
        if "final answer:" in low or "answer:" in low:
            quality["final_answer_like"] += 1
        if "step 1:" in low:
            quality["step_format_like"] += 1

    report = {
        "source_rows": source_stats,
        "unique_pool_by_category": {k: len(v) for k, v in by_cat.items()},
        "auto_selection": auto_report,
        "target_quotas": quotas,
        "curated_rows": len(curated),
        "curated_by_category": dict(Counter(r["category"] for r in curated)),
        "quality": quality,
    }

    return curated, report


def build_combined_test_dataset(seed: int) -> tuple[list[dict], dict]:
    manifest = json.loads((ROOT / "outputs/frozen_eval_manifest.json").read_text(encoding="utf-8"))
    test_rows = load_jsonl(ROOT / "outputs/frozen_test_dataset.jsonl")
    by_id = {str(r.get("id", "")): r for r in test_rows}

    sample_ids = manifest.get("test_sample_ids", {})
    rows: list[dict] = []

    for category, cfg in sample_ids.items():
        ids8 = set(cfg.get("8", []))
        ids20 = list(cfg.get("20", []))
        for tid in ids20:
            base = by_id.get(str(tid))
            if not base:
                continue
            msgs = base.get("messages") or []
            user = ""
            assistant = ""
            for m in msgs:
                role = str(m.get("role", "")).strip().lower()
                content = str(m.get("content", ""))
                if role == "user":
                    user = content
                elif role == "assistant":
                    assistant = content
            rows.append(
                {
                    "id": str(tid),
                    "category": category,
                    "in_fixed8": str(tid) in ids8,
                    "in_fixed20": True,
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ],
                }
            )

    rng = random.Random(seed)
    rng.shuffle(rows)

    report = {
        "rows": len(rows),
        "by_category": dict(Counter(r["category"] for r in rows)),
        "fixed8_rows": sum(1 for r in rows if r["in_fixed8"]),
        "fixed20_rows": sum(1 for r in rows if r["in_fixed20"]),
    }

    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare curated finetune and test datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline-results",
        default="outputs/eval_results_baseline_v5.json",
        help="Baseline evaluation JSON used to auto-compute split/total from weakness metrics.",
    )
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="Optional baseline model id override (defaults to failure_report target model).",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Optional explicit target total examples. If omitted, auto-computed from pool and usage ratio.",
    )
    parser.add_argument("--min-total", type=int, default=700)
    parser.add_argument("--max-total", type=int, default=1400)
    parser.add_argument("--usage-ratio", type=float, default=0.9)
    parser.add_argument("--min-per-category", type=int, default=80)
    parser.add_argument(
        "--source-files",
        nargs="+",
        default=None,
        help="Optional source JSONL files. If omitted, script auto-detects known legacy sources.",
    )
    parser.add_argument(
        "--finetune-output",
        default="finetune/data/curated/finetune_v6_auto.jsonl",
    )
    parser.add_argument(
        "--test-output",
        default="outputs/curated/combined_test_fixed8_20_v2.jsonl",
    )
    parser.add_argument(
        "--report-output",
        default="outputs/curated/curation_report_v2.json",
    )
    args = parser.parse_args()

    if args.source_files:
        source_files = [ROOT / p for p in args.source_files]
    else:
        source_files = _resolve_default_source_files()
    if not source_files:
        raise FileNotFoundError(
            "No source files found. Pass --source-files explicitly or restore known datasets in finetune/data."
        )

    finetune_rows, finetune_report = build_finetune_dataset(
        seed=args.seed,
        source_files=source_files,
        baseline_metrics_path=ROOT / args.baseline_results,
        baseline_model=args.baseline_model,
        target_total=args.target_total,
        min_total=args.min_total,
        max_total=args.max_total,
        usage_ratio=args.usage_ratio,
        min_per_category=args.min_per_category,
    )
    test_rows, test_report = build_combined_test_dataset(seed=args.seed)

    finetune_path = ROOT / args.finetune_output
    test_path = ROOT / args.test_output
    report_path = ROOT / args.report_output

    write_jsonl(finetune_path, finetune_rows)
    write_jsonl(test_path, test_rows)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "finetune": finetune_report,
        "test": test_report,
        "files": {
            "finetune": str(finetune_path.relative_to(ROOT)),
            "test": str(test_path.relative_to(ROOT)),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Prepared datasets:")
    print(f"  Finetune: {finetune_path}")
    print(f"  Test:     {test_path}")
    print(f"  Report:   {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
