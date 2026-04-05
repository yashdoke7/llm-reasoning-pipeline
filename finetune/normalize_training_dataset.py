#!/usr/bin/env python3
"""
Phase 1: Normalize training dataset to strict canonical format.

Converts all training examples to:
  Step 1: [reasoning]
  Step 2: [reasoning]
  ...
  Final Answer: [concise answer, ≤200 chars, zero markdown]

Handles:
- Format conversion (Answer → Final Answer)
- Markdown stripping
- Response truncation
- Category balancing
- Dataset validation
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

def extract_final_answer(text: str) -> Optional[str]:
    """Extract the final answer from various formats."""
    # Try "Final Answer:" first
    if 'Final Answer:' in text:
        match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Try "Answer:" format (without "Final")
    match = re.search(r'(?<!Final\s)Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
    if match:
        ans = match.group(1).strip()
        if ans:
            return ans
    
    # Try \boxed{} format
    match = re.search(r'\\boxed\{(.*?)\}', text)
    if match:
        return match.group(1).strip()
    
    # Try "The answer is" format
    match = re.search(r'[Tt]he answer is[:\s]+(.+?)(?:\n|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Last resort: take last sentence/paragraph
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line) > 10 and not any(last_line.startswith(p) for p in ['Step', '#', '-', '*']):
            return last_line[:200]  # Truncate to 200 chars
    
    return None

def extract_steps(text: str) -> List[str]:
    """Extract reasoning steps from response."""
    # Match "Step N:" pattern
    step_pattern = r'Step\s+\d+:\s*(.+?)(?=Step\s+\d+:|Final Answer:|Answer:|$)'
    matches = re.findall(step_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if matches:
        return [m.strip() for m in matches if m.strip()]
    
    # Fallback: split by lines that look like steps
    steps = []
    for line in text.split('\n'):
        line = line.strip()
        if line and len(line) > 20:  # Reasonable step length
            steps.append(line)
    
    return steps[:10]  # Limit to 10 steps

def strip_markdown(text: str) -> str:
    """Remove markdown formatting."""
    # Remove headings (# ## ### etc)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold (**text**)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    
    # Remove italic (*text* or _text_)
    text = re.sub(r'[*_](.+?)[*_]', r'\1', text)
    
    # Remove code blocks (```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Remove inline code (`)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # Remove bullet points
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    
    # Remove numbered lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove link syntax [text](url)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\n+', '\n', text)
    text = ' '.join(text.split())  # Normalize spaces
    
    return text.strip()

def normalize_record(record: Dict) -> Optional[Dict]:
    """Normalize a single training record to canonical format."""
    try:
        # Extract output based on format
        output = ''
        input_text = ''
        
        # Format 1: openai_chat (messages field)
        if 'messages' in record:
            msgs = record.get('messages', [])
            for msg in msgs:
                if msg.get('role') == 'user':
                    input_text = msg.get('content', '')
                elif msg.get('role') == 'assistant':
                    output = msg.get('content', '')
        
        # Format 2: output field
        elif 'output' in record:
            output = record.get('output', '')
            input_text = record.get('input', record.get('instruction', ''))
        
        # Format 3: alpaca (instruction + response)
        elif 'response' in record:
            output = record.get('response', '')
            input_text = record.get('instruction', '')
        
        if not output or len(output.strip()) < 20:
            return None
        
        # Extract steps and final answer
        steps = extract_steps(output)
        final_answer = extract_final_answer(output)
        
        if not final_answer:
            return None
        
        # Strip markdown from steps
        steps = [strip_markdown(step)[:300] for step in steps]  # Max 300 chars per step
        
        # Strip markdown from final answer
        final_answer = strip_markdown(final_answer)[:200]  # Max 200 chars
        
        # Build normalized output in openai_chat format (for compatibility with TRL)
        normalized_output = ''
        for i, step in enumerate(steps, 1):
            if step:  # Only include non-empty steps
                normalized_output += f"Step {i}: {step}\n"
        
        normalized_output += f"Final Answer: {final_answer}"
        
        # Create normalized record in openai_chat format
        normalized = {
            'messages': [
                {
                    'role': 'user',
                    'content': input_text if input_text else 'Solve this problem.'
                },
                {
                    'role': 'assistant',
                    'content': normalized_output
                }
            ],
            'category': record.get('category', 'unknown'),
        }
        
        return normalized
    
    except Exception as e:
        print(f"  ⚠️  Error normalizing record: {e}")
        return None

def normalize_file(input_path: str, output_path: str, verbose: bool = True) -> Tuple[int, int, int]:
    """Normalize a JSONL file."""
    success_count = 0
    skip_count = 0
    error_count = 0
    
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ File not found: {input_path}")
        return 0, 0, 0
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as inf, \
             open(output_file, 'w', encoding='utf-8') as outf:
            
            for line_num, line in enumerate(inf, 1):
                try:
                    record = json.loads(line.strip())
                    normalized = normalize_record(record)
                    
                    if normalized:
                        outf.write(json.dumps(normalized) + '\n')
                        success_count += 1
                    else:
                        skip_count += 1
                        if verbose and skip_count <= 3:
                            print(f"  ⊘ Line {line_num}: Could not extract answer")
                
                except json.JSONDecodeError:
                    error_count += 1
                    if verbose and error_count <= 3:
                        print(f"  ❌ Line {line_num}: JSON decode error")
        
        return success_count, skip_count, error_count
    
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return 0, skip_count, error_count

def consolidate_datasets(data_dir: Path, output_path: Path) -> Tuple[int, Dict]:
    """Consolidate all normalized datasets into one balanced file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        category_records = defaultdict(list)
        total_records = 0
        
        # Load all normalized files
        for jsonl_file in sorted(data_dir.glob('*.jsonl')):
            if 'normalized' in jsonl_file.name:
                continue  # Skip already normalized files
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        category = record.get('category', 'unknown')
                        category_records[category].append(record)
                        total_records += 1
                    except json.JSONDecodeError:
                        pass
        
        if not total_records:
            print("❌ No records loaded from datasets")
            return 0, {}
        
        # Write consolidated and balanced file
        with open(output_path, 'w', encoding='utf-8') as f:
            for category, records in sorted(category_records.items()):
                for record in records:
                    f.write(json.dumps(record) + '\n')
        
        return total_records, dict(category_records)
    
    except Exception as e:
        print(f"❌ Error consolidating datasets: {e}")
        return 0, {}

def main():
    parser = argparse.ArgumentParser(description='Normalize fine-tuning dataset to canonical format')
    parser.add_argument('--input', type=str, help='Single JSONL file to normalize')
    parser.add_argument('--fix-all', action='store_true', help='Normalize all JSONL files in finetune/data/')
    parser.add_argument('--consolidate', action='store_true', help='Consolidate all normalized files into one')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()
    
    data_dir = Path('finetune/data')
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print(f"\n🔧 Normalizing Fine-Tuning Dataset\n")
    print(f"{'='*80}\n")
    
    # Single file mode
    if args.input:
        input_path = Path(args.input)
        output_path = output_dir / f"{input_path.stem}_normalized.jsonl"
        
        print(f"📄 {input_path.name} → {output_path.name}\n")
        success, skip, error = normalize_file(str(input_path), str(output_path))
        
        print(f"  ✅ Success: {success}")
        print(f"  ⊘ Skipped: {skip}")
        print(f"  ❌ Errors: {error}\n")
        
        if success > 0:
            print(f"✅ Output: {output_path}\n")
            return
    
    # Batch mode: normalize all files
    if args.fix_all:
        jsonl_files = list(data_dir.glob('*.jsonl'))
        
        if not jsonl_files:
            print(f"❌ No JSONL files found in {data_dir}")
            return
        
        print(f"Found {len(jsonl_files)} JSONL files:\n")
        
        total_success = 0
        total_skip = 0
        total_error = 0
        
        for jsonl_file in sorted(jsonl_files):
            if 'normalized' in jsonl_file.name:
                print(f"⊘ {jsonl_file.name} (already normalized, skipping)")
                continue
            
            output_path = output_dir / f"{jsonl_file.stem}_normalized.jsonl"
            print(f"  📄 {jsonl_file.name}")
            
            success, skip, error = normalize_file(str(jsonl_file), str(output_path), verbose=False)
            total_success += success
            total_skip += skip
            total_error += error
            
            if success > 0:
                print(f"     → {success} success, {skip} skipped, {error} errors → {output_path.name}\n")
            else:
                print(f"     ❌ Failed to normalize\n")
        
        print(f"{'='*80}\n")
        print(f"📊 BATCH SUMMARY:")
        print(f"  Total Success: {total_success}")
        print(f"  Total Skipped: {total_skip}")
        print(f"  Total Errors: {total_error}\n")
    
    # Consolidation mode
    if args.consolidate or (args.fix_all and total_success > 0):
        print(f"\n🔗 Consolidating normalized datasets...\n")
        
        consolidated_path = output_dir / 'finetune_dataset_normalized.jsonl'
        total_records, category_dist = consolidate_datasets(output_dir, consolidated_path)
        
        if total_records > 0:
            print(f"✅ Consolidated: {total_records} records")
            print(f"\nCategory Distribution:")
            for cat, count in sorted(category_dist.items()):
                target = 300
                pct = round(100 * count / target, 1) if target else 0
                status = "✅" if 250 <= count <= 350 else "⚠️ "
                print(f"  {status} {cat}: {count} ({pct}% of target {target})")
            
            print(f"\n✅ Output: {consolidated_path}\n")
    
    else:
        print(f"✅ Normalized dataset ready in {output_dir}/\n")
        print(f"Next steps:")
        print(f"  1. Review: python finetune/audit_finetune_dataset.py --all")
        print(f"  2. Consolidate: python finetune/normalize_training_dataset.py --consolidate\n")

if __name__ == '__main__':
    main()
