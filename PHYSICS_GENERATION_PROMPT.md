# Physics Problem Generation Prompt for ChatGPT UI

## Instructions

1. Copy the **MAIN PROMPT** below
2. Paste it into ChatGPT UI (use ChatGPT 4 or 4o for higher quality)
3. Run it once with `count=10` to test output format
4. If satisfied, increase to `count=50` for full dataset
5. Copy the JSON output and save as `physics_problems.json`
6. Run the conversion script below to convert to JSONL format

---

## MAIN PROMPT (Copy-paste entire block into ChatGPT)

```
You are an expert physics tutor. Generate EXACTLY {count} unique physics problems with full step-by-step reasoning traces.

REQUIREMENTS:
- Each problem MUST test either:
  * Physics concept understanding (mechanics, thermodynamics, waves, electromagnetism, optics)
  * Correct formula application
  * Multi-step unit conversions or calculations
  
- STRUCTURE for each problem:
  1. Problem statement (realistic, 1-2 sentences)
  2. Given values with units
  3. What to find
  4. Correct final answer
  5. Complete step-by-step reasoning trace

- REASONING TRACE RULES:
  * Number each step: "Step 1:", "Step 2:", etc.
  * Each step = ONE logical operation (define variable, apply formula, substitute, calculate)
  * Show all unit conversions
  * Show all intermediate calculations with their results
  * End with "Final Answer: [numerical value with units]"
  * MUST be mathematically correct

- DOMAIN DISTRIBUTION (varied topics):
  * 40% Mechanics (motion, forces, energy, momentum, circular motion)
  * 30% Thermodynamics (heat, temperature, gas laws, efficiency)
  * 20% Waves & Optics (sound, light, reflection, refraction, interference)
  * 10% Electromagnetism (electric fields, current, magnetism, Ohm's law)

- DIFFICULTY: Mix of high school and intro university level
- OUTPUT: Return ONLY valid JSON array, no other text

OUTPUT FORMAT (return this exact structure):
[
  {{
    "problem_number": 1,
    "topic": "Mechanics - Kinematics",
    "problem": "A car accelerates uniformly from rest to 25 m/s in 8 seconds. What is the acceleration, and how far does it travel during this time?",
    "given": {{"initial_velocity": "0 m/s", "final_velocity": "25 m/s", "time": "8 s"}},
    "find": "acceleration (a) and distance (d)",
    "reasoning_trace": "Step 1: Identify the kinematic variables: v₀=0 m/s, v=25 m/s, t=8 s.\nStep 2: Use the first kinematic equation v = v₀ + at to find acceleration. Rearrange: a = (v - v₀)/t = (25 - 0)/8 = 3.125 m/s².\nStep 3: Use the second kinematic equation d = v₀*t + (1/2)*a*t² to find distance.\nStep 4: Substitute: d = 0*8 + (1/2)*3.125*8² = 0 + (1/2)*3.125*64 = 1.5625*64 = 100 m.\nFinal Answer: acceleration = 3.125 m/s², distance = 100 m",
    "final_answer": "a = 3.125 m/s, d = 100 m"
  }},
  {{
    "problem_number": 2,
    ...
  }}
]

CRITICAL: Return ONLY the JSON array. No preamble, no markdown code blocks, just raw JSON.
Generate {count} problems now:
```

---

## Step-by-step Instructions for ChatGPT UI

### Step 1: Generate Initial Test (10 problems)
Replace `{count}` with `10` and paste the prompt into ChatGPT. Run once to verify format.

### Step 2: Generate Full Dataset (50 problems)
Replace `{count}` with `50` and paste again. Copy the entire JSON output.

### Step 3: Save JSON Output
1. Create file: `physics_problems.json`
2. Paste the JSON output from ChatGPT
3. Validate JSON syntax (paste into https://jsonlint.com if needed)

### Step 4: Convert to JSONL Format
Run this Python script to convert JSON → JSONL (matching your dataset format):

```python
import json
import sys

def convert_physics_to_finetune_format(input_file, output_file):
    """Convert ChatGPT physics JSON to finetune_dataset.jsonl format"""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    with open(output_file, 'w') as out:
        for problem in data:
            # Match the exact format of your existing finetune_dataset.jsonl
            entry = {
                "text": f"<|system|>\nYou are an expert physics tutor. Generate a perfect, step-by-step reasoning trace for the given problem.\n\nRequirements:\n- Number each step: Step 1:, Step 2:, etc.\n- Each step must contain exactly one logical operation or inference\n- Each step must be explicit, not skipping any computation\n- End with: Final Answer: <answer>\n- The final answer MUST be correct\n- Do not make any factual errors\n<|user|>\nProblem: {problem['problem']}\nGiven: {str(problem['given'])}\nFind: {problem['find']}\n<|assistant|>\n{problem['reasoning_trace']}\n",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert physics tutor. Generate a perfect, step-by-step reasoning trace for the given problem.\n\nRequirements:\n- Number each step: Step 1:, Step 2:, etc.\n- Each step must contain exactly one logical operation or inference\n- Each step must be explicit, not skipping any computation\n- End with: Final Answer: <answer>\n- The final answer MUST be correct\n- Do not make any factual errors"
                    },
                    {
                        "role": "user",
                        "content": f"Problem: {problem['problem']}\nGiven: {str(problem['given'])}\nFind: {problem['find']}"
                    },
                    {
                        "role": "assistant",
                        "content": problem['reasoning_trace']
                    }
                ]
            }
            out.write(json.dumps(entry) + '\n')
    
    print(f"✓ Converted {len(data)} physics problems to {output_file}")
    print(f"  Format: Compatible with finetune_dataset.jsonl")
    print(f"  Total lines: {len(data)}")

if __name__ == "__main__":
    convert_physics_to_finetune_format('physics_problems.json', 'ft_physics.jsonl')
```

Save as: `convert_physics_to_jsonl.py`

Run:
```bash
python convert_physics_to_jsonl.py
```

This produces: `ft_physics.jsonl` (ready for fine-tuning)

---

## Alternative: If ChatGPT Output Needs Formatting

If ChatGPT returns slightly different JSON structure, use this fallback converter:

```python
import json

def flexible_convert(input_file, output_file):
    """Flexible converter that adapts to slight JSON variations"""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    count = 0
    with open(output_file, 'w') as out:
        for problem in data:
            # Extract fields flexibly
            prob_text = problem.get('problem', problem.get('question', ''))
            reasoning = problem.get('reasoning_trace', problem.get('solution', ''))
            
            if not prob_text or not reasoning:
                print(f"⚠ Skipping problem {count+1}: missing required fields")
                continue
            
            entry = {
                "text": f"<|system|>\nYou are an expert physics tutor. Generate a perfect, step-by-step reasoning trace for the given problem.\n\nRequirements:\n- Number each step: Step 1:, Step 2:, etc.\n- Each step must contain exactly one logical operation or inference\n- End with: Final Answer: <answer>\n<|user|>\nProblem: {prob_text}\n<|assistant|>\n{reasoning}\n",
                "messages": [
                    {"role": "system", "content": "You are an expert physics tutor. Generate a perfect, step-by-step reasoning trace for the given problem.\n\nRequirements:\n- Number each step: Step 1:, Step 2:, etc.\n- Each step must contain exactly one logical operation or inference\n- End with: Final Answer: <answer>"},
                    {"role": "user", "content": f"Problem: {prob_text}"},
                    {"role": "assistant", "content": reasoning}
                ]
            }
            out.write(json.dumps(entry) + '\n')
            count += 1
    
    print(f"✓ Converted {count} problems to {output_file}")

flexible_convert('physics_problems.json', 'ft_physics.jsonl')
```

---

## Timeline to Complete

1. **Now**: Copy MAIN PROMPT, test with 10 problems in ChatGPT UI (5 min)
2. **If good**: Generate 50 problems (5 min running)
3. **Save output**: physics_problems.json (2 min)
4. **Convert**: Run conversion script (1 min)
5. **Result**: ft_physics.jsonl ready for fine-tuning (tomorrow morning start)

**Total time: ~20 minutes**

---

## Quality Checklist After Conversion

After generating `ft_physics.jsonl`, verify:
```bash
# Count lines
wc -l ft_physics.jsonl

# Validate JSON each line
python -c "
import json
with open('ft_physics.jsonl') as f:
    count = 0
    for line in f:
        try:
            json.loads(line)
            count += 1
        except:
            print(f'Invalid JSON at line {count+1}')
    print(f'✓ All {count} lines valid JSON')
"
```

Expected: 50+ lines, all valid JSON, ready for training.
