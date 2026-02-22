#!/usr/bin/env python3
"""
Script to systematically improve benchmark task instructions by:
1. Extracting function/class names from test files
2. Inferring signatures and return types from test assertions
3. Generating well-specified instructions
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional

BENCHMARK_DIR = Path("/scratch/10000/eguha3/dc-agent/data/benchmark_tasks_by_dataset")


def extract_function_info_from_test(test_content: str) -> Dict:
    """Extract function names, signatures, and expected behaviors from test file."""
    info = {
        "functions": [],
        "classes": [],
        "assertions": [],
        "imports": []
    }

    # Find function calls in assertions
    # Pattern: function_name(args) == expected or function_name(args).method()
    func_call_pattern = r'assert\s+(\w+)\s*\(([^)]*)\)\s*==\s*([^\n]+)'
    for match in re.finditer(func_call_pattern, test_content):
        func_name = match.group(1)
        args = match.group(2).strip()
        expected = match.group(3).strip()
        info["functions"].append({
            "name": func_name,
            "args": args,
            "expected": expected
        })

    # Find class instantiations
    # Pattern: ClassName(args) or var = ClassName(args)
    class_pattern = r'(\w+)\s*=\s*([A-Z]\w+)\s*\(([^)]*)\)'
    for match in re.finditer(class_pattern, test_content):
        var_name = match.group(1)
        class_name = match.group(2)
        args = match.group(3).strip()
        info["classes"].append({
            "name": class_name,
            "var": var_name,
            "args": args
        })

    # Find method calls on objects
    method_pattern = r'(\w+)\.(\w+)\s*\(([^)]*)\)'
    for match in re.finditer(method_pattern, test_content):
        obj_name = match.group(1)
        method_name = match.group(2)
        args = match.group(3).strip()
        if obj_name not in ['sys', 'os', 'assert']:
            info["assertions"].append({
                "object": obj_name,
                "method": method_name,
                "args": args
            })

    # Find direct function assertions
    direct_assert_pattern = r'assert\s+(\w+)\(([^)]*)\)\s*(?:==|is|!=|in|not)'
    for match in re.finditer(direct_assert_pattern, test_content):
        func_name = match.group(1)
        args = match.group(2).strip()
        if func_name not in [f["name"] for f in info["functions"]]:
            info["functions"].append({
                "name": func_name,
                "args": args,
                "expected": None
            })

    return info


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
        return "None or value"
    elif expected.replace('.', '').replace('-', '').isdigit():
        if '.' in expected:
            return "float"
        return "int"
    else:
        return "value"


def generate_improved_instruction(
    task_name: str,
    original_instruction: str,
    test_info: Dict,
    test_content: str
) -> str:
    """Generate an improved, well-specified instruction."""

    # Extract the title from original instruction
    title_match = re.search(r'# Task: (.+)', original_instruction)
    title = title_match.group(1) if title_match else task_name

    # Build the new instruction
    lines = [f"# Task: {title}", ""]

    # Add function specifications
    if test_info["functions"]:
        for func in test_info["functions"]:
            func_name = func["name"]
            args = func["args"]
            expected = func["expected"]

            # Parse args to get parameter names
            param_names = []
            if args:
                # Simple parsing - split by comma, get names
                for arg in args.split(','):
                    arg = arg.strip()
                    if '=' in arg:
                        param_names.append(arg.split('=')[0].strip())
                    elif arg and not arg.startswith(("'", '"', '[', '{', '(')):
                        # Try to extract a meaningful param name
                        if arg.replace('.', '').replace('-', '').isdigit():
                            param_names.append('n')
                        else:
                            param_names.append(arg.split('[')[0])

            # Create signature
            if param_names:
                sig = f"{func_name}({', '.join(param_names)})"
            else:
                sig = f"{func_name}()"

            return_type = infer_return_type(expected)

            # Get description from original if available
            desc_match = re.search(r'(?:Implement|Create|Write|Fix)[:\s]+(.+?)(?:\.|$)',
                                   original_instruction, re.IGNORECASE)
            if desc_match:
                desc = desc_match.group(1).strip()
            else:
                desc = title.lower()

            lines.append(f"Implement `{sig}` that {desc}.")
            lines.append("")

            # Add return type info
            if expected:
                lines.append(f"**Returns**: `{return_type}` - {expected[:50]}{'...' if len(str(expected)) > 50 else ''}")
                lines.append("")

            # Add example from test
            lines.append("**Example**:")
            lines.append("```python")
            lines.append(f"{func_name}({args}) == {expected}")
            lines.append("```")
            lines.append("")

    # Add class specifications
    if test_info["classes"]:
        for cls in test_info["classes"]:
            class_name = cls["name"]
            args = cls["args"]

            lines.append(f"Implement class `{class_name}`:")
            lines.append("")

            # Find methods called on this class
            methods = []
            for assertion in test_info["assertions"]:
                if assertion["object"] == cls["var"]:
                    method_sig = f"{assertion['method']}({assertion['args']})"
                    if method_sig not in methods:
                        methods.append(method_sig)

            if methods:
                lines.append("**Required methods**:")
                for method in methods:
                    lines.append(f"- `{method}`")
                lines.append("")

    # Standard requirements
    lines.append("## Requirements")
    lines.append("")
    lines.append("- Write your solution in `solution.py`")
    lines.append("- Make sure all tests pass")
    lines.append("")

    return "\n".join(lines)


def improve_task(task_dir: Path) -> Tuple[str, str, bool]:
    """Improve a single task's instruction. Returns (old, new, changed)."""
    instruction_path = task_dir / "instruction.md"
    test_path = task_dir / "tests" / "test_solution.py"

    if not instruction_path.exists() or not test_path.exists():
        return "", "", False

    with open(instruction_path) as f:
        original = f.read()

    with open(test_path) as f:
        test_content = f.read()

    # Check if already well-specified (has backtick function name)
    if re.search(r'`\w+\([^)]*\)`', original):
        return original, original, False

    test_info = extract_function_info_from_test(test_content)

    if not test_info["functions"] and not test_info["classes"]:
        return original, original, False

    improved = generate_improved_instruction(
        task_dir.name, original, test_info, test_content
    )

    return original, improved, True


def improve_dataset(dataset_name: str) -> Dict:
    """Improve all tasks in a dataset."""
    dataset_dir = BENCHMARK_DIR / dataset_name

    if not dataset_dir.exists():
        return {"error": f"Dataset {dataset_name} not found"}

    results = {
        "dataset": dataset_name,
        "total": 0,
        "improved": 0,
        "unchanged": 0,
        "errors": 0,
        "tasks": []
    }

    for task_dir in sorted(dataset_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        results["total"] += 1

        try:
            old, new, changed = improve_task(task_dir)

            if changed:
                results["improved"] += 1
                # Write the improved instruction
                with open(task_dir / "instruction.md", 'w') as f:
                    f.write(new)
                results["tasks"].append({
                    "name": task_dir.name,
                    "status": "improved",
                    "old_preview": old[:100],
                    "new_preview": new[:200]
                })
            else:
                results["unchanged"] += 1
        except Exception as e:
            results["errors"] += 1
            results["tasks"].append({
                "name": task_dir.name,
                "status": "error",
                "error": str(e)
            })

    return results


def main():
    """Improve all datasets."""
    datasets = [
        # Low pass rate - need most improvement
        "ghactions", "bugsinpy_mf", "manybugs", "bugswarm",
        "travistorrent", "e2egit", "defects4j", "softwareheritage",
        "codereval", "crosscodeeval",
        # Medium pass rate
        "codeelo", "bigcodebench", "exercism", "bugsinpy",
        "quixbugs", "swebench",
        # High pass rate - might already be good
        "codenet", "taco", "unitsyn", "methods2test", "pymethods2test"
    ]

    all_results = []

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset}")
        print('='*60)

        results = improve_dataset(dataset)
        all_results.append(results)

        print(f"  Total tasks: {results['total']}")
        print(f"  Improved: {results['improved']}")
        print(f"  Unchanged: {results['unchanged']}")
        print(f"  Errors: {results['errors']}")

        # Show a sample improved task
        for task in results.get("tasks", [])[:2]:
            if task["status"] == "improved":
                print(f"\n  Sample improvement for {task['name']}:")
                print(f"    Old: {task['old_preview'][:80]}...")
                print(f"    New: {task['new_preview'][:120]}...")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_improved = sum(r["improved"] for r in all_results)
    total_tasks = sum(r["total"] for r in all_results)
    print(f"Total tasks improved: {total_improved}/{total_tasks}")

    return all_results


if __name__ == "__main__":
    main()
