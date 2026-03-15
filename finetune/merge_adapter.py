"""
finetune/merge_adapter.py
Merges trained LoRA adapter weights into the base model,
producing a standalone merged model ready for quantization.

Run:
    python finetune/merge_adapter.py
    python finetune/merge_adapter.py --adapter outputs/finetuned_model --output outputs/merged_model
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import yaml

logger = logging.getLogger(__name__)


def merge(
    base_model_id: str,
    adapter_path: str,
    output_path: str,
) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        raise ImportError("Run: pip install transformers peft torch")

    if not torch.cuda.is_available():
        logger.warning("No GPU found — merging on CPU (will be slow for large models)")

    logger.info(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",           # merge on CPU to avoid VRAM issues
        trust_remote_code=True,
    )

    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    logger.info(f"Merge complete. Merged model at: {output_path}")
    logger.info("Next step: run finetune/quantize.py to convert to GGUF")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (default: from config)")
    parser.add_argument("--output", default=None, help="Output path for merged model")
    args = parser.parse_args()

    config_path = _ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    base_model = cfg["finetune"]["base_model"]
    adapter = args.adapter or cfg["finetune"]["output_dir"]
    output = args.output or os.path.join(cfg["paths"]["outputs"], "merged_model")

    merge(base_model, adapter, output)


if __name__ == "__main__":
    main()
