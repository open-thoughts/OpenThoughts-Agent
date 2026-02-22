#!/usr/bin/env python3
"""
Regenerate ghactions benchmark tasks with well-specified instructions.
Self-contained script - no external dependencies.
"""

import re
from pathlib import Path


def extract_function_signatures_from_test(test_code: str) -> dict:
    """Extract function/class names and signatures from test code."""
    result = {
        "functions": [],
        "classes": [],
    }

    builtins = ['len', 'str', 'int', 'float', 'list', 'dict', 'set', 'type', 'isinstance', 'sorted', 'any', 'all']

    # Find function calls in assertions: assert func_name(args) == expected
    func_assert_pattern = r'assert\s+(\w+)\s*\(([^)]*)\)\s*==\s*([^\n]+)'
    for match in re.finditer(func_assert_pattern, test_code):
        func_name = match.group(1)
        args = match.group(2).strip()
        expected = match.group(3).strip()

        if func_name in builtins:
            continue

        result["functions"].append({
            "name": func_name,
            "args": args,
            "expected": expected
        })

    # Find wrapped patterns: assert len(func(args)) == expected
    wrapped_pattern = r'assert\s+(len|str|int|float|sorted)\s*\(\s*(\w+)\s*\(([^)]*)\)\s*\)\s*==\s*([^\n]+)'
    for match in re.finditer(wrapped_pattern, test_code):
        wrapper = match.group(1)
        func_name = match.group(2)
        args = match.group(3).strip()
        expected = match.group(4).strip()

        if func_name in builtins:
            continue

        result["functions"].append({
            "name": func_name,
            "args": args,
            "expected": f"{wrapper}(...) == {expected}",
            "returns": "list" if wrapper == "len" else wrapper
        })

    # Find function calls with 'is' comparisons
    is_pattern = r'assert\s+(\w+)\s*\(([^)]*)\)\s+is\s+(not\s+)?(\w+)'
    for match in re.finditer(is_pattern, test_code):
        func_name = match.group(1)
        args = match.group(2).strip()
        negated = match.group(3)
        expected = match.group(4)

        if func_name in ['len', 'str', 'int', 'float', 'list', 'dict', 'set', 'type', 'isinstance']:
            continue

        result["functions"].append({
            "name": func_name,
            "args": args,
            "expected": f"{'not ' if negated else ''}{expected}"
        })

    # Find dict/attribute access patterns: assert func(args)['key'] == value
    dict_access_pattern = r'assert\s+(\w+)\s*\(([^)]*)\)\s*\[([^\]]+)\]\s*==\s*([^\n]+)'
    for match in re.finditer(dict_access_pattern, test_code):
        func_name = match.group(1)
        args = match.group(2).strip()
        key = match.group(3).strip()
        expected = match.group(4).strip()

        if func_name in ['len', 'str', 'int', 'float', 'list', 'dict', 'set', 'type', 'isinstance']:
            continue

        # Check if already found
        if not any(f["name"] == func_name for f in result["functions"]):
            result["functions"].append({
                "name": func_name,
                "args": args,
                "expected": f"dict with [{key}] == {expected}",
                "returns_dict": True,
                "key_accessed": key
            })

    # Find class instantiations
    class_pattern = r'(\w+)\s*=\s*([A-Z]\w+)\s*\(([^)]*)\)'
    for match in re.finditer(class_pattern, test_code):
        var_name = match.group(1)
        class_name = match.group(2)
        args = match.group(3).strip()

        method_pattern = rf'{var_name}\.(\w+)\s*\(([^)]*)\)'
        methods = []
        for method_match in re.finditer(method_pattern, test_code):
            method_name = method_match.group(1)
            method_args = method_match.group(2).strip()
            methods.append({
                "name": method_name,
                "args": method_args
            })

        result["classes"].append({
            "name": class_name,
            "init_args": args,
            "methods": methods
        })

    # Deduplicate functions by name
    seen_funcs = set()
    unique_funcs = []
    for f in result["functions"]:
        if f["name"] not in seen_funcs:
            seen_funcs.add(f["name"])
            unique_funcs.append(f)
    result["functions"] = unique_funcs

    return result


def generate_well_specified_instruction(test_code: str, task_title: str = "Task") -> str:
    """Generate a well-specified instruction from test code."""
    extracted = extract_function_signatures_from_test(test_code)

    lines = [f"# Task: {task_title}", ""]

    # Add functions
    for func in extracted["functions"]:
        func_name = func["name"]
        args = func["args"]
        expected = func["expected"]

        # Infer parameter names
        param_names = []
        if args:
            for i, arg in enumerate(args.split(',')):
                arg = arg.strip()
                if arg.startswith(("'", '"', '[', '{', '(')):
                    param_names.append(f"arg{i+1}")
                elif arg.replace('.', '').replace('-', '').replace('+', '').isdigit():
                    param_names.append(f"n" if i == 0 else f"arg{i+1}")
                else:
                    clean_arg = arg.split('[')[0].split('.')[0].strip()
                    if clean_arg.isidentifier():
                        param_names.append(clean_arg)
                    else:
                        param_names.append(f"arg{i+1}")

        sig = f"{func_name}({', '.join(param_names)})" if param_names else f"{func_name}()"

        # Determine return type hint
        if func.get("returns_dict"):
            return_hint = "dict"
        elif expected.startswith('['):
            return_hint = "list"
        elif expected.startswith('{'):
            return_hint = "dict"
        elif expected in ['True', 'False']:
            return_hint = "bool"
        elif expected.replace('.', '').replace('-', '').isdigit():
            return_hint = "int" if '.' not in expected else "float"
        else:
            return_hint = "value"

        lines.append(f"Implement `{sig}` that returns a `{return_hint}`.")
        lines.append("")

        if expected:
            lines.append("**Example**:")
            lines.append("```python")
            lines.append(f"{func_name}({args}) == {expected}")
            lines.append("```")
            lines.append("")

    # Add classes
    for cls in extracted["classes"]:
        class_name = cls["name"]
        init_args = cls["init_args"]
        methods = cls["methods"]

        lines.append(f"Implement class `{class_name}`:")
        lines.append("")

        if init_args:
            lines.append(f"- Constructor: `__init__({init_args})`")

        for method in methods:
            method_sig = f"{method['name']}({method['args']})"
            lines.append(f"- Method: `{method_sig}`")

        lines.append("")

    # Requirements
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
    """Regenerate all tasks in a dataset."""
    results = {
        "total": 0,
        "improved": 0,
        "already_specified": 0,
        "no_signatures": 0,
        "samples": []
    }

    for task_dir in sorted(dataset_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        results["total"] += 1

        instruction_path = task_dir / "instruction.md"
        test_path = task_dir / "tests" / "test_solution.py"

        if not instruction_path.exists() or not test_path.exists():
            continue

        old_instruction = instruction_path.read_text(encoding="utf-8")
        test_code = test_path.read_text(encoding="utf-8")

        # Force regenerate all - don't skip any
        # (Previous attempts may have created garbled instructions)

        # Extract and generate
        extracted = extract_function_signatures_from_test(test_code)

        if not extracted["functions"] and not extracted["classes"]:
            results["no_signatures"] += 1
            continue

        title = get_task_title(task_dir.name)
        new_instruction = generate_well_specified_instruction(test_code, title)

        # Write new instruction
        instruction_path.write_text(new_instruction, encoding="utf-8")
        results["improved"] += 1

        # Store sample for display
        if len(results["samples"]) < 5:
            results["samples"].append({
                "task": task_dir.name,
                "functions": [f["name"] for f in extracted["functions"]],
                "old": old_instruction[:100],
                "new": new_instruction[:200]
            })

    return results


def main():
    dataset_name = "ghactions"
    dataset_dir = Path(f"/scratch/10000/eguha3/dc-agent/data/benchmark_tasks_by_dataset/{dataset_name}")

    print(f"Regenerating {dataset_name} benchmark tasks...")
    print("=" * 60)

    results = regenerate_dataset(dataset_dir)

    print(f"Total tasks: {results['total']}")
    print(f"Improved: {results['improved']}")
    print(f"Already specified: {results['already_specified']}")
    print(f"No signatures found: {results['no_signatures']}")

    print("\nSample improvements:")
    for sample in results["samples"]:
        print(f"\n  Task: {sample['task']}")
        print(f"  Functions found: {sample['functions']}")
        print(f"  Old: {sample['old'][:60]}...")
        print(f"  New: {sample['new'][:100]}...")


if __name__ == "__main__":
    main()
