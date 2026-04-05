"""
Build frozen train/validation/test splits from the normalized training dataset.

This creates stable, reproducible splits and a manifest for exact benchmark
samples so future runs use the same tasks instead of random slices.

Default input:
    outputs/finetune_dataset_normalized.jsonl

Outputs:
    outputs/frozen_train_dataset.jsonl
    outputs/frozen_val_dataset.jsonl
    outputs/frozen_test_dataset.jsonl
    outputs/frozen_eval_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _stable_id(row: dict, fallback_index: int) -> str:
    if row.get("id"):
        return str(row["id"])
    if row.get("messages"):
        for msg in row["messages"]:
            if msg.get("role") == "user" and msg.get("content"):
                text = str(msg["content"]).strip().replace("\n", " ")
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
                return f"row_{fallback_index:06d}_{digest}"
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"row_{fallback_index:06d}_{digest}"


def _split_bucket(sample_id: str, salt: str) -> int:
    digest = hashlib.sha1(f"{salt}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _split_name(sample_id: str, salt: str) -> str:
    bucket = _split_bucket(sample_id, salt)
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_manifest(test_rows: list[dict], salt: str) -> dict:
    by_cat: dict[str, list[str]] = defaultdict(list)
    for idx, row in enumerate(test_rows):
        by_cat[row.get("category", "unknown")].append(_stable_id(row, idx))

    manifest = {
        "salt": salt,
        "sample_order": "sorted_by_id",
        "test_sample_ids": {},
    }

    for cat, ids in sorted(by_cat.items()):
        ids = sorted(ids)
        manifest["test_sample_ids"][cat] = {
            "8": ids[:8],
            "20": ids[:20],
        }

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen train/val/test splits")
    parser.add_argument("--input", default="outputs/finetune_dataset_normalized.jsonl")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--salt", default="llm-reasoning-frozen-v1")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    rows = _read_jsonl(input_path)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for idx, row in enumerate(rows):
        sid = _stable_id(row, idx)
        split = _split_name(sid, args.salt)
        row_with_id = dict(row)
        row_with_id.setdefault("id", sid)
        if split == "train":
            train_rows.append(row_with_id)
        elif split == "val":
            val_rows.append(row_with_id)
        else:
            test_rows.append(row_with_id)

    for bucket in (train_rows, val_rows, test_rows):
        bucket.sort(key=lambda row: _stable_id(row, 0))

    _write_jsonl(output_dir / "frozen_train_dataset.jsonl", train_rows)
    _write_jsonl(output_dir / "frozen_val_dataset.jsonl", val_rows)
    _write_jsonl(output_dir / "frozen_test_dataset.jsonl", test_rows)

    manifest = _build_manifest(test_rows, args.salt)
    manifest.update(
        {
            "source": str(input_path),
            "counts": {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
                "total": len(rows),
            },
        }
    )
    with open(output_dir / "frozen_eval_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Frozen splits created:")
    print(f"  train: {len(train_rows)}")
    print(f"  val:   {len(val_rows)}")
    print(f"  test:  {len(test_rows)}")
    print(f"  manifest: {output_dir / 'frozen_eval_manifest.json'}")


if __name__ == "__main__":
    main()
