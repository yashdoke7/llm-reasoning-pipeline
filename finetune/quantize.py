"""
finetune/quantize.py
Converts a merged HuggingFace model to GGUF format using llama.cpp.
Produces Q4_K_M and Q8_0 quantized versions for benchmarking.

Prerequisites:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make -j$(nproc)
    pip install -r requirements.txt   (inside llama.cpp directory)

Then set LLAMA_CPP_DIR env var or pass --llama-cpp-dir

Run:
    export LLAMA_CPP_DIR=/path/to/llama.cpp
    python finetune/quantize.py
    python finetune/quantize.py --model outputs/merged_model --formats Q4_K_M Q8_0
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import yaml

logger = logging.getLogger(__name__)


def _find_llama_cpp(provided: str | None) -> str | None:
    """Locate llama.cpp directory from env var, argument, or common locations."""
    candidates = [
        provided,
        os.environ.get("LLAMA_CPP_DIR"),
        os.path.expanduser("~/llama.cpp"),
        "/opt/llama.cpp",
        str(_ROOT / "llama.cpp"),
    ]
    for c in candidates:
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "convert_hf_to_gguf.py")):
            return c
    return None


def convert_to_fp16_gguf(
    model_path: str,
    llama_cpp_dir: str,
    output_dir: str,
) -> str:
    """Step 1: Convert HF model to GGUF FP16 (lossless intermediate)."""
    os.makedirs(output_dir, exist_ok=True)
    gguf_fp16_path = os.path.join(output_dir, "model_fp16.gguf")

    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    cmd = [
        sys.executable,
        convert_script,
        model_path,
        "--outfile", gguf_fp16_path,
        "--outtype", "f16",
    ]
    logger.info(f"Converting to FP16 GGUF: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed:\n{result.stderr}")
    logger.info(f"FP16 GGUF saved: {gguf_fp16_path}")
    return gguf_fp16_path


def quantize_gguf(
    fp16_path: str,
    quant_type: str,
    output_dir: str,
    llama_cpp_dir: str,
) -> str:
    """Step 2: Quantize GGUF to target quantization level."""
    quantize_bin = os.path.join(llama_cpp_dir, "llama-quantize")
    if not os.path.exists(quantize_bin):
        # Try alternative name
        quantize_bin = os.path.join(llama_cpp_dir, "quantize")
    if not os.path.exists(quantize_bin):
        raise FileNotFoundError(
            f"llama-quantize binary not found in {llama_cpp_dir}. "
            "Run 'make' inside the llama.cpp directory."
        )

    out_path = os.path.join(output_dir, f"model_{quant_type.lower()}.gguf")
    cmd = [quantize_bin, fp16_path, out_path, quant_type]

    logger.info(f"Quantizing to {quant_type}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed for {quant_type}:\n{result.stderr}")

    size_mb = os.path.getsize(out_path) / 1e6
    logger.info(f"{quant_type} GGUF saved: {out_path} ({size_mb:.1f} MB)")
    return out_path


def run_quantization(
    model_path: str,
    quant_formats: list[str],
    output_dir: str,
    llama_cpp_dir: str | None = None,
) -> dict[str, str]:
    """
    Full quantization pipeline: HF -> FP16 GGUF -> quantized GGUFs.

    Returns:
        dict mapping quant_type -> output_path
    """
    llama_dir = _find_llama_cpp(llama_cpp_dir)
    if not llama_dir:
        raise FileNotFoundError(
            "llama.cpp not found. Clone it: git clone https://github.com/ggerganov/llama.cpp\n"
            "Then: cd llama.cpp && make -j$(nproc)\n"
            "Then: export LLAMA_CPP_DIR=/path/to/llama.cpp"
        )
    logger.info(f"Using llama.cpp at: {llama_dir}")

    # Convert to FP16 first
    fp16_path = convert_to_fp16_gguf(model_path, llama_dir, output_dir)

    # Quantize to each format
    results = {"fp16": fp16_path}
    for quant_type in quant_formats:
        try:
            out = quantize_gguf(fp16_path, quant_type, output_dir, llama_dir)
            results[quant_type] = out
        except RuntimeError as e:
            logger.error(f"Failed to quantize {quant_type}: {e}")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("QUANTIZATION SUMMARY")
    logger.info("=" * 50)
    for fmt, path in results.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            logger.info(f"  {fmt:12s}  {size_mb:8.1f} MB  {path}")

    logger.info("\nNext step: run experiments/run_comparison_eval.py to benchmark these models")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize fine-tuned model to GGUF")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None, help="Path to merged HF model")
    parser.add_argument("--formats", nargs="+", default=None, help="Quantization formats (e.g. Q4_K_M Q8_0)")
    parser.add_argument("--output", default=None, help="Output directory for GGUF files")
    parser.add_argument("--llama-cpp-dir", default=None, help="Path to llama.cpp directory")
    args = parser.parse_args()

    config_path = _ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    model_path = args.model or os.path.join(cfg["paths"]["outputs"], "merged_model")
    formats = args.formats or [f["name"] for f in cfg["finetune"]["quantization"]["formats"]]
    output = args.output or cfg["finetune"]["quantization"]["output_dir"]

    run_quantization(
        model_path=model_path,
        quant_formats=formats,
        output_dir=output,
        llama_cpp_dir=args.llama_cpp_dir,
    )


if __name__ == "__main__":
    main()
