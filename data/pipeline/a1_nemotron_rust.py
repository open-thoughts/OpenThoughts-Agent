#!/usr/bin/env python3
"""
Generate Rust test tasks from Nemotron - filters for self-contained Rust test files.
Only accepts tests using standard library - no external crate dependencies.
"""

import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from nemotron_loader import load_nemotron_stream, normalize_sample, extract_code_from_sample
from data.completions import run_completions
from data.commons import create_standard_task_toml, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10_000
MODEL = "gpt-5-nano-2025-08-07"  # Updated v4 - reject inner attributes

RUST_TEST_TO_TASK_PROMPT = """You are an expert at creating coding tasks from Rust test code.

Given the following Rust test code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/src/lib.rs

Test code:
{{text}}

Create a task description (just the task, no preamble):"""

# Standard library modules that are safe to use
RUST_STD_MODULES = {
    'std', 'core', 'alloc',
    # Common std modules
    'collections', 'env', 'error', 'ffi', 'fmt', 'fs', 'hash', 'io', 'iter',
    'marker', 'mem', 'net', 'num', 'ops', 'option', 'path', 'process',
    'result', 'slice', 'str', 'string', 'sync', 'thread', 'time', 'vec',
    'cell', 'rc', 'borrow', 'clone', 'cmp', 'convert', 'default', 'any',
    'ascii', 'char', 'f32', 'f64', 'i8', 'i16', 'i32', 'i64', 'i128', 'isize',
    'u8', 'u16', 'u32', 'u64', 'u128', 'usize', 'bool', 'panic', 'ptr',
    'array', 'hint', 'intrinsics', 'primitive', 'task', 'future', 'pin',
}


def is_rust_test_file(content: str) -> bool:
    """Check if content is a self-contained Rust test file using only std."""
    has_test_attr = '#[test]' in content
    has_cfg_test = '#[cfg(test)]' in content
    has_assert = 'assert!' in content or 'assert_eq!' in content or 'assert_ne!' in content

    if not ((has_test_attr or has_cfg_test) and has_assert):
        return False

    # Reject files that use crate:: imports (depend on external library code)
    if 'use crate::' in content or 'crate::' in content:
        return False

    # Reject files with extern crate (depend on external crates)
    if 'extern crate' in content:
        return False

    # Reject files with super:: imports (depend on parent modules)
    if 'use super::' in content or 'super::' in content:
        return False

    # Reject files that use unstable features
    if '#![feature' in content or '#[feature' in content:
        return False

    # Reject files using unstable core/std internals
    if 'core::num::' in content or 'std::num::' in content:
        return False

    # Reject files with inner attributes (e.g., #![allow(...)]) - they must be at file start
    # which conflicts with our prepended "use solution::*;"
    if '#![' in content:
        return False

    # Reject files that use test macros (test_vfs!, etc)
    if re.search(r'\b\w+!\s*\(.*\)', content) and not re.search(r'(assert|println|print|format|vec|panic|debug_assert|write|writeln|todo|unimplemented|unreachable)!', content):
        # Check if there are unknown macros being used
        known_macros = {'assert', 'assert_eq', 'assert_ne', 'println', 'print', 'format',
                        'vec', 'panic', 'debug_assert', 'debug_assert_eq', 'debug_assert_ne',
                        'write', 'writeln', 'todo', 'unimplemented', 'unreachable', 'cfg',
                        'derive', 'test', 'should_panic', 'ignore', 'allow', 'warn', 'deny',
                        'forbid', 'deprecated', 'must_use', 'doc', 'inline', 'repr', 'feature'}
        macro_uses = re.findall(r'\b(\w+)!\s*[\(\[\{]', content)
        for macro in macro_uses:
            if macro.lower() not in known_macros:
                return False

    # Extract all use statements and check they only use std
    # Match both "use foo::bar" and "use foo as alias" and "use foo;"
    use_statements = re.findall(r'use\s+(\w+)(?:::|;|\s+as\s+)', content)
    for mod in use_statements:
        if mod not in RUST_STD_MODULES:
            return False

    return True


def count_test_functions(content: str) -> int:
    """Count number of #[test] functions."""
    return len(re.findall(r'#\[test\]', content))


def filter_rust_tests_from_nemotron(limit: int, max_scan: int = 50_000_000) -> List[Dict]:
    """Filter Nemotron for Rust test content."""
    print("Loading Nemotron (Rust subset, streaming)...")
    ds = load_nemotron_stream("rust")

    rust_samples = []
    scanned = 0

    print(f"Scanning for Rust test files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding Rust test files")

    for raw_sample in itertools.islice(ds, max_scan):
        scanned += 1

        # CC-Code-v1: extract code blocks from web pages
        if "text" in raw_sample and "content" not in raw_sample:
            content = extract_code_from_sample(raw_sample, "rust")
            if not content:
                continue
            sample = {"content": content, "path": raw_sample.get("uuid", ""), "repo": "cc-code-v1", "size": len(content)}
        else:
            sample = normalize_sample(raw_sample)
            content = sample['content']

        if is_rust_test_file(content):
            test_count = count_test_functions(content)
            if test_count >= 2:
                rust_samples.append({
                    'text': content,
                    'path': sample['path'],
                    'repo': sample['repo'],
                    'size': sample['size'],
                    'test_count': test_count,
                })
                pbar.update(1)

                if len(rust_samples) >= limit:
                    break

        if scanned % 50000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(rust_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(rust_samples)} Rust test files")
    return rust_samples


def create_rust_dockerfile() -> str:
    """Create Rust Dockerfile for test environment."""
    return """FROM rust:1.82-slim

WORKDIR /app

# Install build tools and bash for Harbor agent
RUN apt-get update && apt-get install -y \\
    pkg-config \\
    libssl-dev \\
    bash \\
    && rm -rf /var/lib/apt/lists/*

# Pre-create Cargo project structure
RUN cargo init --name solution --lib && \\
    mkdir -p src tests
"""


def create_test_sh() -> str:
    """Create test.sh that runs Rust tests and reports results."""
    return '''#!/bin/bash
set -e

mkdir -p /logs/verifier

cleanup() {
    if [ $? -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
    else
        echo "0" > /logs/verifier/reward.txt
    fi
}
trap cleanup EXIT

cd /app

# Initialize cargo project if needed
if [ ! -f Cargo.toml ]; then
    cargo init --name solution --lib
fi

# Ensure src directory and lib.rs exist
mkdir -p src
if [ ! -f src/lib.rs ]; then
    touch src/lib.rs
fi

# Copy test file to tests directory as integration test
mkdir -p tests
cp /tests/test_solution.rs tests/

# Prepend "use solution::*;" to test file so it can access the library
# Only if it doesn't already have a use solution statement
if ! grep -q "use solution::" tests/test_solution.rs; then
    # Create temp file with use statement + original content
    echo 'use solution::*;' > tests/test_solution_tmp.rs
    cat tests/test_solution.rs >> tests/test_solution_tmp.rs
    mv tests/test_solution_tmp.rs tests/test_solution.rs
fi

# Run tests
echo "Running Rust tests..."
timeout 120 cargo test 2>&1 | tee /logs/verifier/test_output.txt

exit ${PIPESTATUS[0]}
'''


def create_harbor_task_directory(
    output_dir: Path,
    task_id: int,
    instruction_content: str,
    test_code: str,
    dataset_prefix: str,
    metadata: Optional[Dict] = None,
) -> Path:
    """Create a harbor-format task directory."""
    task_dir = output_dir / f"{dataset_prefix}-{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(create_rust_dockerfile(), encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh(), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)
    (tests_dir / "test_solution.rs").write_text(test_code, encoding="utf-8")

    (task_dir / "instruction.md").write_text(instruction_content, encoding="utf-8")
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    if metadata is None:
        metadata = {}
    metadata["source_dataset"] = "nemotron"
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_rust_tasks(samples: List[Dict], task_descriptions: List[str], dataset_prefix: str = "nemotron-rust") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, description) in enumerate(tqdm(zip(samples, task_descriptions), total=len(samples), desc="Creating tasks")):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
            "test_count": sample.get("test_count", 0),
        }
        create_harbor_task_directory(temp_dir, i, description, sample["text"], dataset_prefix, metadata)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main() -> None:
    """Main pipeline."""
    print("Step 1: Filtering Rust test files from Nemotron...")
    rust_samples = filter_rust_tests_from_nemotron(LIMIT)
    print(f"  -> {len(rust_samples)} Rust test files found")

    if not rust_samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Synthesizing tasks...")
    dataset = Dataset.from_list(rust_samples)
    result = run_completions(
        dataset, model=MODEL, map_type="chat",
        map_config={"user_message": RUST_TEST_TO_TASK_PROMPT, "output_column": "task_description"},
        max_requests_per_minute=500, max_tokens_per_minute=1_000_000,
        require_all_responses=False,  # Allow partial results for files exceeding context length
    )
    task_descriptions = result.dataset["task_description"]
    print(f"  -> Generated {len(task_descriptions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_rust_tasks(rust_samples, task_descriptions, "nemotron-rust")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_nemotron-rust")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(rust_samples)} Rust tasks!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
