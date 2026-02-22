#!/usr/bin/env python3
"""
Regenerate ALL benchmark tasks with well-specified instructions.
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

    builtins = ['len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
                'type', 'isinstance', 'sorted', 'any', 'all', 'sum', 'min', 'max',
                'abs', 'round', 'range', 'enumerate', 'zip', 'map', 'filter', 'print']

    # Pattern 1: assert func(args) == expected
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

    # Pattern 2: assert len(func(args)) == expected (wrapped functions)
    wrapped_pattern = r'assert\s+(len|str|int|float|sorted|list|set)\s*\(\s*(\w+)\s*\(([^)]*)\)\s*\)\s*==\s*([^\n]+)'
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
            "expected": f"{wrapper}(result) == {expected}",
            "returns": "list" if wrapper == "len" else "value"
        })

    # Pattern 3: assert func(args) is [not] None/True/False
    is_pattern = r'assert\s+(\w+)\s*\(([^)]*)\)\s+is\s+(not\s+)?(\w+)'
    for match in re.finditer(is_pattern, test_code):
        func_name = match.group(1)
        args = match.group(2).strip()
        negated = match.group(3)
        expected = match.group(4)

        if func_name in builtins:
            continue

        result["functions"].append({
            "name": func_name,
            "args": args,
            "expected": f"{'not ' if negated else ''}{expected}"
        })

    # Pattern 4: assert func(args)['key'] == value (dict access)
    dict_access_pattern = r'assert\s+(\w+)\s*\(([^)]*)\)\s*\[([^\]]+)\]\s*==\s*([^\n]+)'
    for match in re.finditer(dict_access_pattern, test_code):
        func_name = match.group(1)
        args = match.group(2).strip()
        key = match.group(3).strip()
        expected = match.group(4).strip()

        if func_name in builtins:
            continue

        if not any(f["name"] == func_name for f in result["functions"]):
            result["functions"].append({
                "name": func_name,
                "args": args,
                "expected": f"dict with [{key}] == {expected}",
                "returns": "dict"
            })

    # Pattern 5: assert func(args).attr == value (attribute access)
    attr_pattern = r'assert\s+(\w+)\s*\(([^)]*)\)\.(\w+)\s*==\s*([^\n]+)'
    for match in re.finditer(attr_pattern, test_code):
        func_name = match.group(1)
        args = match.group(2).strip()
        attr = match.group(3).strip()
        expected = match.group(4).strip()

        if func_name in builtins:
            continue

        if not any(f["name"] == func_name for f in result["functions"]):
            result["functions"].append({
                "name": func_name,
                "args": args,
                "expected": f"object with .{attr} == {expected}",
                "returns": "object"
            })

    # Pattern 6: var = func(args); assert ... (multi-line)
    call_pattern = r'(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)'
    for match in re.finditer(call_pattern, test_code):
        var_name = match.group(1)
        func_name = match.group(2)
        args = match.group(3).strip()

        if func_name in builtins:
            continue
        if func_name[0].isupper():  # Likely a class
            continue

        if not any(f["name"] == func_name for f in result["functions"]):
            result["functions"].append({
                "name": func_name,
                "args": args,
                "expected": None
            })

    # Pattern 7: Class instantiation
    class_pattern = r'(\w+)\s*=\s*([A-Z]\w+)\s*\(([^)]*)\)'
    for match in re.finditer(class_pattern, test_code):
        var_name = match.group(1)
        class_name = match.group(2)
        args = match.group(3).strip()

        # Find method calls on this instance
        method_pattern = rf'{var_name}\.(\w+)\s*\(([^)]*)\)'
        methods = []
        for method_match in re.finditer(method_pattern, test_code):
            method_name = method_match.group(1)
            method_args = method_match.group(2).strip()
            if method_name not in ['__init__', '__str__', '__repr__']:
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


def infer_return_type(expected: str) -> str:
    """Infer return type from expected value."""
    if expected is None:
        return "value"
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
        if '.' in expected:
            return "float"
        return "int"
    elif 'dict' in expected.lower():
        return "dict"
    elif 'list' in expected.lower() or 'len' in expected.lower():
        return "list"
    else:
        return "value"


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
                    param_names.append("n" if i == 0 else f"arg{i+1}")
                else:
                    clean_arg = arg.split('[')[0].split('.')[0].strip()
                    if clean_arg.isidentifier() and len(clean_arg) < 20:
                        param_names.append(clean_arg)
                    else:
                        param_names.append(f"arg{i+1}")

        sig = f"{func_name}({', '.join(param_names)})" if param_names else f"{func_name}()"

        # Determine return type
        ret_type = func.get("returns") or infer_return_type(expected)

        lines.append(f"Implement `{sig}` that returns a `{ret_type}`.")
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
            lines.append(f"- Constructor: `__init__(self, {init_args})`")
        else:
            lines.append(f"- Constructor: `__init__(self)`")

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
        "no_signatures": 0,
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
        if len(results["samples"]) < 3:
            results["samples"].append({
                "task": task_dir.name,
                "functions": [f["name"] for f in extracted["functions"]],
                "classes": [c["name"] for c in extracted["classes"]],
                "old": old_instruction[:80],
                "new": new_instruction[:150]
            })

    return results


def main():
    benchmark_dir = Path("/scratch/10000/eguha3/dc-agent/data/benchmark_tasks_by_dataset")

    # All 21 datasets
    datasets = [
        # 0% pass rate
        "ghactions", "bugsinpy_mf", "manybugs", "bugswarm",
        # Low pass rate
        "travistorrent", "e2egit", "defects4j", "softwareheritage",
        "codereval", "crosscodeeval",
        # Medium pass rate
        "codeelo", "bigcodebench", "exercism", "bugsinpy",
        "quixbugs", "swebench",
        # High pass rate
        "pymethods2test", "methods2test", "unitsyn", "taco", "codenet",
    ]

    print("=" * 70)
    print("REGENERATING ALL BENCHMARK TASKS WITH WELL-SPECIFIED INSTRUCTIONS")
    print("=" * 70)

    total_improved = 0
    total_tasks = 0
    all_results = []

    for dataset in datasets:
        dataset_dir = benchmark_dir / dataset

        if not dataset_dir.exists():
            print(f"\n[SKIP] {dataset}: directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {dataset}")
        print('='*60)

        results = regenerate_dataset(dataset_dir)
        all_results.append((dataset, results))

        total_improved += results["improved"]
        total_tasks += results["total"]

        print(f"  Total: {results['total']}, Improved: {results['improved']}, "
              f"No signatures: {results['no_signatures']}, Errors: {results['errors']}")

        # Show samples
        for sample in results["samples"][:2]:
            print(f"\n  {sample['task']}:")
            print(f"    Functions: {sample['functions']}")
            if sample['classes']:
                print(f"    Classes: {sample['classes']}")
            print(f"    Old: {sample['old'][:50]}...")
            print(f"    New: {sample['new'][:70]}...")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal tasks improved: {total_improved}/{total_tasks} "
          f"({100*total_improved/total_tasks:.1f}%)")

    print("\nPer-dataset breakdown:")
    print(f"{'Dataset':<20} {'Total':>6} {'Improved':>10} {'Rate':>8}")
    print("-" * 50)
    for dataset, results in all_results:
        rate = 100 * results["improved"] / results["total"] if results["total"] > 0 else 0
        print(f"{dataset:<20} {results['total']:>6} {results['improved']:>10} {rate:>7.1f}%")


if __name__ == "__main__":
    main()
