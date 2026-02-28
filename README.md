# LLM Reasoning Evaluation + LoRA Fine-Tuning Pipeline

Step-level evaluation of LLM reasoning chains with targeted LoRA fine-tuning and quantization benchmarking.

---

## What This Does

Most LLM benchmarks grade only the final answer. This pipeline grades **every intermediate reasoning step** — finding where models fail, why they fail, and using that data to fine-tune more effectively.

**Phase 1 (CPU / Free API):** Evaluate 3 open-source LLMs across 4 reasoning categories. Generate step-level failure analysis and hallucination reports.

**Phase 2 (GPU):** Use the failure report to curate targeted fine-tuning data. Fine-tune with QLoRA. Quantize to Q4/Q8 GGUF. Re-evaluate using the same harness to measure improvement.

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo>
cd llm-reasoning-pipeline
pip install -r requirements.txt
```

### 2. Set API keys

```bash
export GROQ_API_KEY=your_groq_key       # Get free at: console.groq.com
export WANDB_API_KEY=your_wandb_key     # Get free at: wandb.ai
```

### 3. (Optional) Download CRASS dataset

The CRASS dataset is used for counterfactual reasoning evaluation.
Without it, the pipeline uses a built-in synthetic fallback (10 examples).

```
1. Go to: https://github.com/apergo-ai/CRASS
2. Download the file: data/crass_dataset.csv
3. Place it at: datasets/data/crass_dataset.csv
```

---

## Running Phase 1 (Evaluation)

### Quick test (3 models, 10 samples per category)

```bash
python experiments/run_baseline_eval.py --samples 10 --no-wandb
```

### Full evaluation (recommended)

```bash
python experiments/run_baseline_eval.py
```

### Single model, single category

```bash
python experiments/run_baseline_eval.py \
  --models llama-3-8b-8192 \
  --categories multistep_arithmetic \
  --samples 50
```

### Without mitigation (faster, for initial exploration)

```bash
python experiments/run_baseline_eval.py --no-mitigation --samples 20
```

### Launch the Streamlit demo

```bash
streamlit run dashboard/app.py
```

---

## Running Phase 2 (Fine-Tuning)

**Requires:** RTX 4070 (8GB VRAM) or better, PyTorch with CUDA

### Install GPU dependencies

```bash
pip install -r requirements-finetune.txt
```

### Step 1: Build fine-tuning dataset

Reads the failure report from Phase 1 and generates training traces using Mixtral.

```bash
python finetune/dataset_builder.py --samples 3000
```

### Step 2: Fine-tune with LoRA

```bash
python finetune/train_lora.py
```

### Step 3: Merge LoRA adapter into base model

```bash
python finetune/merge_adapter.py
```

### Step 4: Quantize to GGUF

First, install llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j$(nproc)
export LLAMA_CPP_DIR=$(pwd)
cd ..
```

Then quantize:

```bash
python finetune/quantize.py
```

### Step 5: Compare baseline vs fine-tuned vs quantized

```bash
python experiments/run_comparison_eval.py
```

---

## Project Structure

```
llm-reasoning-pipeline/
├── configs/config.yaml          # All settings — edit this first
├── models/groq_client.py        # Groq API client with retry + rate limiting
├── eval/
│   ├── task_decomposer.py       # Builds prompts, parses CoT traces into steps
│   ├── step_evaluator.py        # Step-level meta-evaluation (VALID/INVALID/UNCERTAIN)
│   ├── hallucination_scorer.py  # Flags unverifiable factual claims
│   ├── drift_detector.py        # Detects contradictions and topic drift
│   └── metrics.py               # Aggregation + W&B logging
├── mitigation/
│   ├── retriever.py             # Wikipedia retriever for RAG
│   └── regrounder.py            # Re-grounds failing steps with retrieved context
├── finetune/
│   ├── dataset_builder.py       # Builds training data from failure cases
│   ├── train_lora.py            # QLoRA training (PEFT + bitsandbytes + TRL)
│   ├── merge_adapter.py         # Merges LoRA into base model
│   └── quantize.py              # Exports to GGUF via llama.cpp
├── datasets/
│   ├── gsm8k_loader.py          # Auto-downloads from HuggingFace
│   ├── crass_loader.py          # CRASS + synthetic fallback
│   ├── toolbench_loader.py      # Synthetic tool-use tasks
│   └── factual_synthetic.py     # Synthetic factual consistency tasks
├── experiments/
│   ├── run_baseline_eval.py     # Phase 1: full benchmark
│   └── run_comparison_eval.py   # Phase 2: baseline vs fine-tuned comparison
├── dashboard/app.py             # Streamlit UI
├── outputs/                     # Results, reports, charts (auto-created)
├── logs/                        # Log files (auto-created)
├── requirements.txt
└── requirements-finetune.txt
```

---

## Datasets

| Dataset | Category | Auto-download? | Notes |
|---------|----------|---------------|-------|
| GSM8K | Multi-step arithmetic | ✅ Yes (HuggingFace) | 8,500 math problems |
| CRASS | Causal/counterfactual | ❌ Manual (GitHub) | 285 samples; synthetic fallback included |
| Tool-use | Tool planning | ✅ Yes (synthetic) | 8 built-in tasks; extend as needed |
| Factual | Factual consistency | ✅ Yes (synthetic) | 8 built-in topics; auto-cached |

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| Step failure rate | % of reasoning steps graded INVALID |
| Error propagation rate | % of step failures that caused wrong final answers |
| Hallucination rate | % of tasks with ≥1 hallucination-flagged step |
| Final accuracy | % of tasks with correct final answer |
| Mitigation delta | Accuracy improvement from RAG re-grounding |

---

## Configuration

All settings are in `configs/config.yaml`. Key things to change:

- `eval.samples_per_category`: Start with 20-50 for testing, use 150+ for full eval
- `models.eval_models`: Add/remove models (must be available on Groq)
- `finetune.base_model`: Change to `meta-llama/Meta-Llama-3-8B-Instruct` for 8B (needs more VRAM)
- `api.wandb_project`: Set to your W&B project name

---

## Resume Bullet

> Built a step-level LLM reasoning evaluation harness benchmarking 3 open-source models (Llama-3, Mixtral, Gemma-2) across 4 reasoning categories; used failure analysis to guide domain LoRA fine-tuning on Llama-3 with QLoRA, reducing step-failure rate by X%; benchmarked accuracy-efficiency tradeoffs across Q4/Q8 quantization levels.
