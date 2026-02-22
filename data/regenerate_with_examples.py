#!/usr/bin/env python3
"""
Regenerate benchmark tasks with ACTUAL examples from test code.
Key improvement: Include real input values, not variable names.
"""

import re
from pathlib import Path


def extract_test_examples(test_code: str) -> list:
    """Extract actual test examples with real values from test code."""
    examples = []

    # Find variable assignments followed by assertions
    # Pattern: var = 'value' or var = [...] followed by assert func(var)
    var_assignments = {}
    for match in re.finditer(r"(\w+)\s*=\s*(['\"][^'\"]+['\"]|\[[^\]]+\]|\{[^\}]+\}|[0-9.-]+)", test_code):
        var_name = match.group(1)
        var_value = match.group(2)
        var_assignments[var_name] = var_value

    # Pattern 1: assert func(var) == expected where var is defined above
    for match in re.finditer(r'assert\s+(\w+)\s*\(([^)]*)\)\s*==\s*([^\n]+)', test_code):
        func_name = match.group(1)
        args = match.group(2).strip()
        expected = match.group(3).strip()

        if func_name in ['len', 'str', 'int', 'float', 'list', 'sorted', 'set', 'isinstance']:
            continue

        # Substitute variable names with actual values
        resolved_args = args
        for var_name, var_value in var_assignments.items():
            resolved_args = re.sub(rf'\b{var_name}\b', var_value, resolved_args)

        examples.append({
            'func_name': func_name,
            'args': args,
            'resolved_args': resolved_args,
            'expected': expected,
            'has_real_values': resolved_args != args or "'" in resolved_args or '"' in resolved_args or '[' in resolved_args
        })

    # Pattern 2: assert func(var)['key'] == expected (dict access)
    for match in re.finditer(r"assert\s+(\w+)\s*\(([^)]*)\)\s*\[(['\"][^'\"]+['\"])\]\s*==\s*([^\n]+)", test_code):
        func_name = match.group(1)
        args = match.group(2).strip()
        key = match.group(3).strip()
        expected = match.group(4).strip()

        if func_name in ['len', 'str', 'int', 'float', 'list', 'sorted', 'set', 'isinstance']:
            continue

        # Substitute variable names with actual values
        resolved_args = args
        for var_name, var_value in var_assignments.items():
            resolved_args = re.sub(rf'\b{var_name}\b', var_value, resolved_args)

        # Check if not already captured
        if not any(e['func_name'] == func_name for e in examples):
            examples.append({
                'func_name': func_name,
                'args': args,
                'resolved_args': resolved_args,
                'expected': f"dict with [{key}] == {expected}",
                'has_real_values': resolved_args != args or "'" in resolved_args or '"' in resolved_args,
                'returns_dict': True
            })

    # Pattern 3: assert func(var).attr == expected (attribute access)
    for match in re.finditer(r"assert\s+(\w+)\s*\(([^)]*)\)\.(\w+)\s*==\s*([^\n]+)", test_code):
        func_name = match.group(1)
        args = match.group(2).strip()
        attr = match.group(3).strip()
        expected = match.group(4).strip()

        if func_name in ['len', 'str', 'int', 'float', 'list', 'sorted', 'set', 'isinstance']:
            continue

        # Substitute variable names with actual values
        resolved_args = args
        for var_name, var_value in var_assignments.items():
            resolved_args = re.sub(rf'\b{var_name}\b', var_value, resolved_args)

        # Check if not already captured
        if not any(e['func_name'] == func_name for e in examples):
            examples.append({
                'func_name': func_name,
                'args': args,
                'resolved_args': resolved_args,
                'expected': f"object with .{attr} == {expected}",
                'has_real_values': resolved_args != args or "'" in resolved_args or '"' in resolved_args,
                'returns_object': True
            })

    # Pattern 4: Direct assertions with literal values
    for match in re.finditer(r"assert\s+(\w+)\s*\((['\"\[\{0-9].*?)\)\s*==\s*([^\n]+)", test_code):
        func_name = match.group(1)
        args = match.group(2).strip()
        expected = match.group(3).strip()

        if func_name in ['len', 'str', 'int', 'float', 'list', 'sorted', 'set', 'isinstance']:
            continue

        # Check if not already captured
        if not any(e['func_name'] == func_name and e['args'] == args for e in examples):
            examples.append({
                'func_name': func_name,
                'args': args,
                'resolved_args': args,
                'expected': expected,
                'has_real_values': True
            })

    return examples


def infer_param_names(args: str, test_code: str) -> list:
    """Infer meaningful parameter names from test context."""
    params = []

    # Look for variable assignments that might hint at parameter meaning
    for i, arg in enumerate(args.split(',')):
        arg = arg.strip()

        # If it's a variable name, try to find its meaning
        if arg.isidentifier():
            # Check if there's a pattern like: arg = 'string with hint'
            match = re.search(rf'{arg}\s*=\s*[\'"]([^\'\"]+)[\'"]', test_code)
            if match:
                value = match.group(1)
                if 'error' in value.lower() or 'warning' in value.lower():
                    params.append('log_text')
                elif 'feat:' in value or 'fix:' in value:
                    params.append('commit_msg')
                elif ':' in value and '\n' in value:
                    params.append('yaml_str')
                else:
                    params.append(arg)
            else:
                params.append(arg)
        elif arg.startswith('"') or arg.startswith("'"):
            # String literal - infer from content
            content = arg.strip("'\"")
            if 'error' in content.lower() or 'warning' in content.lower():
                params.append('log_text')
            elif ':' in content:
                params.append('text')
            else:
                params.append('s')
        elif arg.startswith('['):
            params.append('items')
        elif arg.startswith('{'):
            params.append('data')
        elif arg.replace('.', '').replace('-', '').isdigit():
            params.append('n')
        else:
            params.append(f'arg{i+1}')

    return params


def infer_return_type(expected: str) -> str:
    """Infer return type from expected value."""
    expected = expected.strip()

    if expected.startswith('['):
        return "list"
    elif expected.startswith('{'):
        if ':' in expected:
            return "dict"
        return "set"
    elif expected.startswith('('):
        return "tuple"
    elif expected.startswith('"') or expected.startswith("'"):
        return "str"
    elif expected in ['True', 'False']:
        return "bool"
    elif expected == 'None':
        return "Optional[value]"
    elif expected.replace('.', '').replace('-', '').replace('+', '').isdigit():
        return "float" if '.' in expected else "int"
    else:
        return "value"


def generate_improved_instruction(test_code: str, task_title: str) -> str:
    """Generate instruction with actual example values."""
    examples = extract_test_examples(test_code)

    if not examples:
        return None

    lines = [f"# Task: {task_title}", ""]

    # Get unique functions
    seen_funcs = set()
    for ex in examples:
        if ex['func_name'] in seen_funcs:
            continue
        seen_funcs.add(ex['func_name'])

        func_name = ex['func_name']
        params = infer_param_names(ex['args'], test_code)

        # Determine return type based on example
        if ex.get('returns_dict'):
            ret_type = "dict"
        elif ex.get('returns_object'):
            ret_type = "object"
        else:
            ret_type = infer_return_type(ex['expected'])

        sig = f"{func_name}({', '.join(params)})"
        lines.append(f"Implement `{sig}` that returns a `{ret_type}`.")
        lines.append("")

        # Find best example for this function (prefer ones with real values)
        best_examples = [e for e in examples if e['func_name'] == func_name and e['has_real_values']]
        if not best_examples:
            best_examples = [e for e in examples if e['func_name'] == func_name]

        if best_examples:
            lines.append("**Example**:")
            lines.append("```python")
            # Use resolved args (with actual values substituted)
            ex = best_examples[0]
            lines.append(f"{func_name}({ex['resolved_args']}) => {ex['expected']}")
            lines.append("```")
            lines.append("")

    lines.append("## Requirements")
    lines.append("")
    lines.append("- Write your solution in `solution.py`")
    lines.append("- Make sure all tests pass")
    lines.append("")

    return "\n".join(lines)


def get_task_title(dirname: str) -> str:
    """Extract title from directory name."""
    parts = dirname.split('_')
    if len(parts) >= 3:
        title_parts = parts[2:]
        return ' '.join(word.capitalize() for word in title_parts)
    return dirname.replace('_', ' ').title()


def regenerate_dataset(dataset_dir: Path) -> dict:
    """Regenerate all tasks in a dataset with improved examples."""
    results = {
        "total": 0,
        "improved": 0,
        "unchanged": 0,
        "errors": 0,
        "samples": []
    }

    for task_dir in sorted(dataset_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        results["total"] += 1

        instruction_path = task_dir / "instruction.md"
        test_path = task_dir / "tests" / "test_solution.py"

        if not instruction_path.exists() or not test_path.exists():
            results["errors"] += 1
            continue

        try:
            old_instruction = instruction_path.read_text(encoding="utf-8")
            test_code = test_path.read_text(encoding="utf-8")
        except Exception as e:
            results["errors"] += 1
            continue

        title = get_task_title(task_dir.name)
        new_instruction = generate_improved_instruction(test_code, title)

        if new_instruction is None:
            results["unchanged"] += 1
            continue

        # Check if it's actually improved (has real values in examples)
        if "'" in new_instruction or '"' in new_instruction or '[' in new_instruction:
            instruction_path.write_text(new_instruction, encoding="utf-8")
            results["improved"] += 1

            if len(results["samples"]) < 3:
                results["samples"].append({
                    "task": task_dir.name,
                    "old": old_instruction[:100],
                    "new": new_instruction[:200]
                })
        else:
            results["unchanged"] += 1

    return results


def main():
    benchmark_dir = Path("/scratch/10000/eguha3/dc-agent/data/benchmark_tasks_by_dataset")

    # Focus on low-performing datasets that need better examples
    datasets = [
        "ghactions", "travistorrent", "e2egit", "softwareheritage",
        "manybugs", "bugswarm", "bugsinpy_mf", "defects4j",
        "codereval", "crosscodeeval", "codeelo"
    ]

    print("=" * 70)
    print("REGENERATING WITH ACTUAL EXAMPLE VALUES")
    print("=" * 70)

    total_improved = 0
    total_tasks = 0

    for dataset in datasets:
        dataset_dir = benchmark_dir / dataset

        if not dataset_dir.exists():
            print(f"\n[SKIP] {dataset}: directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {dataset}")
        print('='*60)

        results = regenerate_dataset(dataset_dir)

        total_improved += results["improved"]
        total_tasks += results["total"]

        print(f"  Total: {results['total']}, Improved: {results['improved']}, "
              f"Unchanged: {results['unchanged']}, Errors: {results['errors']}")

        for sample in results["samples"][:2]:
            print(f"\n  {sample['task']}:")
            print(f"    Old: {sample['old'][:60]}...")
            print(f"    New: {sample['new'][:80]}...")

    print("\n" + "=" * 70)
    print(f"TOTAL IMPROVED: {total_improved}/{total_tasks}")
    print("=" * 70)


if __name__ == "__main__":
    main()
