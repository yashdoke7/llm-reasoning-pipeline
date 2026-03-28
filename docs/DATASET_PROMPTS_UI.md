# UI Prompts For Dataset Generation

Use this when generating datasets in ChatGPT (GPT-4.1) or Gemini (3.1 Pro) UI.

## Workflow
1. Paste one prompt below in the UI.
2. Ask for output as a single valid JSON array only (no markdown).
3. Save output to `data_loaders/data/ui_<name>.json`.
4. Convert to JSONL:

```powershell
venv\Scripts\python.exe data_loaders\convert_ui_json_to_jsonl.py --input data_loaders\data\ui_factual.json --category factual_consistency --output data_loaders\data\generated_factual.jsonl
venv\Scripts\python.exe data_loaders\convert_ui_json_to_jsonl.py --input data_loaders\data\ui_tool.json --category tool_use_planning --output data_loaders\data\generated_tooluse.jsonl
venv\Scripts\python.exe data_loaders\convert_ui_json_to_jsonl.py --input data_loaders\data\ui_counterfactual.json --category causal_counterfactual --output data_loaders\data\generated_counterfactual.jsonl
venv\Scripts\python.exe data_loaders\convert_ui_json_to_jsonl.py --input data_loaders\data\ui_physics_eval.json --category physics_reasoning --output data_loaders\data\physics_eval.jsonl
```

## Prompt: Factual Consistency
Generate exactly 80 factual-consistency benchmark tasks as a JSON array.
Return JSON only.

Each item must contain:
- `topic` (string)
- `context` (4-8 sentences with concrete facts)
- `questions` (array of 3-5 questions answerable ONLY from context)
- `answers` (array matching question count, short exact answers)

Rules:
- Diverse domains (history, science, biology, economics, computing, geography, etc.)
- Include specific dates/numbers/names where appropriate.
- No outside-knowledge questions.
- High precision and no duplicate tasks.

## Prompt: Tool-Use Planning
Generate exactly 80 tool-use planning tasks as a JSON array.
Return JSON only.

Each item must contain:
- `goal` (string)
- `available_tools` (array of tool objects: `{name, description, params}`)
- `constraints` (array of strings)
- `correct_plan` (array of 2-5 explicit tool calls)

Allowed tool names:
`search_web`, `read_file`, `write_file`, `send_email`, `run_python`, `query_database`, `download_url`, `summarize_text`, `translate_text`, `compute_stats`, `parse_pdf`, `create_chart`, `schedule_task`, `api_call`.

Rules:
- Steps must be executable and logically ordered.
- Parameter values must be explicit.
- Constraints must be checkable.
- No duplicates.

## Prompt: Causal Counterfactual
Generate exactly 80 causal-counterfactual reasoning tasks as a JSON array.
Return JSON only.

Each item must contain:
- `premise` (counterfactual premise)
- `question` (asks consequences)
- `correct_answer` (1-3 sentences)
- `distractors` (array of 3 plausible but wrong answers)

Rules:
- Domains: science, tech, history, social systems, economics, biology.
- Avoid trivial examples.
- Ensure correct answer follows causally from premise.
- No duplicates.

## Prompt: Physics Eval Set (Holdout)
Generate exactly 60 physics reasoning evaluation tasks as a JSON array.
Return JSON only.

Each item must contain:
- `question` (full problem text)
- `ground_truth` (final answer with units)

Rules:
- Mix mechanics, electricity, fluids, thermodynamics, waves/optics, modern physics.
- Multi-step quantitative reasoning.
- Include units and physically realistic values.
- No overlap with training set.
- No derivation in `ground_truth`, only final answer.

## Prompt: Physics Training Set (Curated CoT)
Generate exactly 160 physics training examples as a JSON array.
Return JSON only.

Each item must contain:
- `problem` (full physics problem)
- `reasoning_trace` (Step 1:, Step 2:, ... and end with `Final Answer:`)
- `final_answer` (short final answer with units)

Rules:
- Reasoning trace must be correct and explicit.
- Include unit checks where relevant.
- Keep one logical action per step.
- No overlap with evaluation tasks.
