# End-to-End Execution Playbook (General + Physics)

This playbook is designed for your April 1 submission workflow.

## 0) Python Command (important)
If `venv\Scripts\python.exe` fails with `Unable to create process`, use:

```powershell
$PY="C:\Path\To\Python311\python.exe"
```

Then run commands as `& $PY ...`.

## 1) Backup Current Results (keep last working run)
```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path "outputs\backup_$stamp" | Out-Null
Copy-Item outputs\*.json "outputs\backup_$stamp\" -Force
```

## 2) Optional: Regenerate Non-Arithmetic Eval Datasets
### Option A: Automated generation script
```powershell
& $PY data_loaders\generate_eval_datasets.py --provider ollama --model qwen2.5:14b --samples 80
```

### Option B: UI generation (GPT-4.1 / Gemini 3.1 Pro)
Use prompts from `docs/DATASET_PROMPTS_UI.md`, then convert:

```powershell
& $PY data_loaders\convert_ui_json_to_jsonl.py --input data_loaders\data\ui_factual.json --category factual_consistency --output data_loaders\data\generated_factual.jsonl
& $PY data_loaders\convert_ui_json_to_jsonl.py --input data_loaders\data\ui_tool.json --category tool_use_planning --output data_loaders\data\generated_tooluse.jsonl
& $PY data_loaders\convert_ui_json_to_jsonl.py --input data_loaders\data\ui_counterfactual.json --category causal_counterfactual --output data_loaders\data\generated_counterfactual.jsonl
```

## 3) Run Fresh Baseline (General)
```powershell
& $PY experiments\run_baseline_eval.py --provider ollama --models qwen2.5:3b --judge-provider ollama --judge-model qwen2.5:14b --samples 50 --no-wandb --mitigation-metrics none
```

This writes:
- `outputs/eval_results_<suffix>.json`
- `outputs/raw_results_<suffix>.json`
- `outputs/failure_report_<suffix>.json`
- plus latest aliases (`eval_results.json`, `raw_results.json`, etc.)

## 4) Build Fine-Tune Dataset (General)
### Recommended (new): failure-driven hybrid
```powershell
& $PY finetune\dataset_builder.py --strategy hybrid --samples 1400 --categories multistep_arithmetic factual_consistency tool_use_planning causal_counterfactual --provider ollama --trace-model qwen2.5:14b --quality-provider ollama --quality-model qwen2.5:14b --output data_loaders\data\finetune_dataset.jsonl
```

### Pure failure-targeted
```powershell
& $PY finetune\dataset_builder.py --strategy failure --samples 1200 --provider ollama --trace-model qwen2.5:14b --quality-provider ollama --quality-model qwen2.5:14b --output data_loaders\data\finetune_dataset.jsonl
```

### Audit mix
```powershell
& $PY finetune\audit_finetune_dataset.py --dataset data_loaders\data\finetune_dataset.jsonl
```

## 5) Fine-Tune + Merge + Quantize
```powershell
& $PY finetune\train_lora.py
& $PY finetune\merge_adapter.py
& $PY finetune\quantize.py --formats Q4_K_M
```

If `Q4_K_M` quantization binary is unavailable, FP16 GGUF is still produced and can be benchmarked.

## 6) Compare Base vs Fine-Tuned (General, all categories)
```powershell
& $PY experiments\run_comparison_eval.py --categories multistep_arithmetic factual_consistency tool_use_planning causal_counterfactual --samples 30 --base-model qwen2.5:3b --base-type ollama --judge-provider ollama --judge-model qwen2.5:14b --with-diagnostics --no-wandb
```

Outputs now include timestamped files + latest aliases:
- `outputs/comparison_results_<suffix>.json` (or category-specific variant)
- `outputs/comparison_raw_<suffix>.json`
- `outputs/comparison_manifest_<suffix>.json`
- plus latest aliases

## 7) Physics-Specific Track
### 7.1 Prepare physics train/eval data from curated JSON
Assume `physics_problems.json` contains `problem`, `reasoning_trace`, `final_answer`.

```powershell
& $PY finetune\prepare_physics_dataset.py --input physics_problems.json --train-output data_loaders\data\ft_physics.jsonl --eval-output data_loaders\data\physics_eval.jsonl --train-ratio 0.8
```

### 7.2 Train physics-specialized model
```powershell
& $PY finetune\train_lora.py --config configs\config.yaml
```

Before running, set `finetune.dataset_path` in config to `data_loaders/data/ft_physics.jsonl`
or pass your own workflow that points training to that file.

### 7.3 Physics benchmark comparison
```powershell
& $PY experiments\run_comparison_eval.py --custom-dataset data_loaders\data\physics_eval.jsonl --custom-category physics_reasoning --samples 40 --base-model qwen2.5:3b --base-type ollama --judge-provider ollama --judge-model qwen2.5:14b --no-wandb
```

## 8) Groq-Limited Validation (small sample sanity check)
```powershell
& $PY experiments\run_comparison_eval.py --categories multistep_arithmetic causal_counterfactual --samples 10 --base-model qwen2.5:3b --base-type ollama --judge-provider groq --judge-model llama-3.3-70b-versatile --no-wandb
```

## 9) Charts
```powershell
& $PY experiments\generate_charts.py --latest-only
```

## 10) Suggested Reporting Structure
1. Baseline (fresh run)
2. General fine-tuned (failure-driven hybrid)
3. Physics-specialized fine-tuned
4. Per-category deltas + physics holdout deltas
5. Mention sample-size constraints for Groq/TPD explicitly
