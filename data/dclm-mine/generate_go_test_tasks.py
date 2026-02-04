#!/usr/bin/env python3
"""
Generate Go test tasks from The Stack - filters for self-contained Go test files.
Only accepts tests using standard library - no external dependencies.
"""

import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import create_standard_task_toml, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10_000
MODEL = "gpt-5-nano-2025-08-07"

GO_TEST_TO_TASK_PROMPT = """You are an expert at creating coding tasks from Go test code.

Given the following Go test code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/

Test code:
{{text}}

Create a task description (just the task, no preamble):"""

# Go standard library packages (safe to use without external deps)
GO_STD_PACKAGES = {
    # Core packages
    'testing', 'fmt', 'errors', 'strings', 'strconv', 'bytes', 'bufio',
    'io', 'os', 'path', 'filepath', 'sort', 'math', 'rand', 'time',
    'unicode', 'utf8', 'utf16', 'regexp', 'reflect', 'unsafe',
    # Collections
    'container/heap', 'container/list', 'container/ring',
    # Encoding
    'encoding', 'encoding/json', 'encoding/xml', 'encoding/base64',
    'encoding/hex', 'encoding/binary', 'encoding/csv', 'encoding/gob',
    'encoding/pem', 'encoding/ascii85', 'encoding/asn1',
    # Compression
    'compress/gzip', 'compress/zlib', 'compress/flate', 'compress/bzip2', 'compress/lzw',
    # Crypto
    'crypto', 'crypto/md5', 'crypto/sha1', 'crypto/sha256', 'crypto/sha512',
    'crypto/hmac', 'crypto/rand', 'crypto/aes', 'crypto/cipher', 'crypto/subtle',
    'crypto/rsa', 'crypto/ecdsa', 'crypto/ed25519', 'crypto/elliptic', 'crypto/x509',
    # Hash
    'hash', 'hash/crc32', 'hash/crc64', 'hash/adler32', 'hash/fnv',
    # Network (simple)
    'net', 'net/url', 'net/http', 'net/http/httptest',
    # Text
    'text/template', 'text/scanner', 'text/tabwriter',
    'html', 'html/template',
    # Sync
    'sync', 'sync/atomic',
    # Context
    'context',
    # Runtime
    'runtime', 'runtime/debug',
    # Image
    'image', 'image/png', 'image/jpeg', 'image/gif', 'image/color',
    # Log
    'log',
    # Flag
    'flag',
    # Archive
    'archive/tar', 'archive/zip',
    # Debug
    'debug/buildinfo', 'debug/dwarf', 'debug/elf', 'debug/gosym', 'debug/macho', 'debug/pe', 'debug/plan9obj',
    # Embed
    'embed',
    # Go parser (useful for code tools)
    'go/ast', 'go/doc', 'go/format', 'go/parser', 'go/printer', 'go/scanner', 'go/token', 'go/types',
    # Database (interface only)
    'database/sql', 'database/sql/driver',
    # Testing
    'testing/iotest', 'testing/quick', 'testing/fstest',
    # Index
    'index/suffixarray',
    # Internal (usually not imported directly)
    'internal',
    # MIME
    'mime', 'mime/multipart', 'mime/quotedprintable',
    # Maps and slices (Go 1.21+)
    'maps', 'slices', 'cmp',
    # Plugin
    'plugin',
    # Expvar
    'expvar',
}


def has_external_imports(content: str) -> bool:
    """Check if Go file imports external (non-stdlib) packages."""
    # Find all import statements
    # Handle both single imports: import "pkg"
    # And grouped imports: import ( "pkg1" \n "pkg2" )

    # Extract all quoted import paths
    import_paths = re.findall(r'"([^"]+)"', content)

    for path in import_paths:
        # Skip empty paths
        if not path:
            continue

        # External packages have domains (github.com, golang.org/x, etc.)
        if '.' in path.split('/')[0]:
            return True

        # Check if it's a standard library package
        # Standard packages don't have dots in the first segment
        base_pkg = path.split('/')[0]
        full_path = path

        # It's external if it looks like a domain
        if any(domain in path for domain in ['github.com', 'gitlab.com', 'bitbucket.org',
                                              'golang.org', 'gopkg.in', 'k8s.io',
                                              'go.uber.org', 'google.golang.org',
                                              'cloud.google.com', 'gonum.org']):
            return True

        # If the package path contains a dot in the first segment, it's external
        if '.' in base_pkg:
            return True

    return False


def is_self_contained_test(content: str) -> bool:
    """Check if test file is self-contained (no imports of project packages)."""
    # Reject tests that import from the same project using relative imports
    # These patterns indicate the test depends on project-specific code

    # Check for relative imports (same package)
    lines = content.split('\n')
    in_import_block = False

    for line in lines:
        stripped = line.strip()

        # Track if we're in an import block
        if stripped.startswith('import ('):
            in_import_block = True
            continue
        if in_import_block and stripped == ')':
            in_import_block = False
            continue

        # Check for problematic import patterns
        if in_import_block or stripped.startswith('import '):
            # Skip named imports that reference current package
            if re.search(r'\.\s+"', stripped):  # dot import
                return False
            if re.search(r'_\s+"', stripped):  # blank import of project code
                # Allow blank imports of standard library
                path_match = re.search(r'_\s+"([^"]+)"', stripped)
                if path_match:
                    path = path_match.group(1)
                    if '.' in path.split('/')[0]:
                        return False

    return True


def is_go_test_file(content: str) -> bool:
    """Check if content is a self-contained Go test file using only stdlib."""
    has_testing_import = 'import "testing"' in content or '"testing"' in content
    has_test_func = bool(re.search(r'func\s+Test\w+\s*\(\s*t\s+\*testing\.T\s*\)', content))
    has_error_calls = 't.Error' in content or 't.Fatal' in content or 't.Fail' in content

    if not (has_testing_import and has_test_func):
        return False

    # Reject tests with external dependencies
    if has_external_imports(content):
        return False

    # Reject tests that depend on project-specific code
    if not is_self_contained_test(content):
        return False

    # Reject tests that are too complex (importing many packages)
    import_count = len(re.findall(r'"[^"]+"', content))
    if import_count > 10:
        return False

    # Prefer shorter, simpler test files
    line_count = len(content.split('\n'))
    if line_count > 300:
        return False

    return True


def count_test_functions(content: str) -> int:
    """Count number of Test functions."""
    return len(re.findall(r'func\s+Test\w+\s*\(', content))


def filter_go_tests_from_stack(limit: int, max_scan: int = 5_000_000) -> List[Dict]:
    """Filter The Stack for Go test content."""
    print("Loading The Stack (Go subset, streaming)...")
    ds = load_dataset(
        'bigcode/the-stack',
        data_dir='data/go',
        split='train',
        streaming=True
    )

    go_samples = []
    scanned = 0

    print(f"Scanning for Go test files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding Go test files")

    for sample in itertools.islice(ds, max_scan):
        scanned += 1
        content = sample.get('content', '')
        path = sample.get('path', '')

        # Check content directly (path may be empty in The Stack)
        if is_go_test_file(content):
            test_count = count_test_functions(content)
            if test_count >= 2:
                go_samples.append({
                    'text': content,
                    'path': path,
                    'repo': sample.get('repository_name', ''),
                    'size': sample.get('size', 0),
                    'test_count': test_count,
                })
                pbar.update(1)

                if len(go_samples) >= limit:
                    break

        if scanned % 50000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(go_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(go_samples)} Go test files")
    return go_samples


def create_go_dockerfile() -> str:
    """Create Go Dockerfile for test environment."""
    return """FROM golang:1.22-alpine

WORKDIR /app

# Install bash (required by Harbor agent), git, build tools for cgo,
# and util-linux for 'script' command (needed by terminus agent)
RUN apk add --no-cache bash git gcc musl-dev util-linux
"""


def create_test_sh() -> str:
    """Create test.sh that runs Go tests and reports results."""
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

# Initialize go module if needed
if [ ! -f go.mod ]; then
    go mod init solution
fi

# Copy test file and fix package name to match the solution package
cp /tests/solution_test.go .

# Replace the package declaration with "package solution" to avoid conflicts
# The test file package must match the solution file package
sed -i '1,5s/^package .*/package solution/' solution_test.go

# Also fix package in any Go files the agent created
for f in *.go; do
    if [ "$f" != "solution_test.go" ] && [ -f "$f" ]; then
        sed -i '1,5s/^package .*/package solution/' "$f"
    fi
done

# Resolve dependencies
go mod tidy 2>&1 || true
go mod download 2>&1 || true

# Run tests
echo "Running Go tests..."
go test -v ./... 2>&1 | tee /logs/verifier/test_output.txt

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
    (env_dir / "Dockerfile").write_text(create_go_dockerfile(), encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh(), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)
    (tests_dir / "solution_test.go").write_text(test_code, encoding="utf-8")

    (task_dir / "instruction.md").write_text(instruction_content, encoding="utf-8")
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    if metadata:
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_go_tasks(samples: List[Dict], task_descriptions: List[str], dataset_prefix: str = "stack-go") -> str:
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
    print("Step 1: Filtering Go test files from The Stack...")
    go_samples = filter_go_tests_from_stack(LIMIT)
    print(f"  -> {len(go_samples)} Go test files found")

    if not go_samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Synthesizing tasks...")
    dataset = Dataset.from_list(go_samples)
    result = run_completions(
        dataset, model=MODEL, map_type="chat",
        map_config={"user_message": GO_TEST_TO_TASK_PROMPT, "output_column": "task_description"},
        max_requests_per_minute=500, max_tokens_per_minute=1_000_000,
        require_all_responses=False,  # Allow some failures for large files
    )
    task_descriptions = result.dataset["task_description"]
    print(f"  -> Generated {len(task_descriptions)} task descriptions")

    # Filter out failed tasks (None values)
    valid_pairs = [(s, d) for s, d in zip(go_samples, task_descriptions) if d is not None]
    if len(valid_pairs) < len(go_samples):
        print(f"  -> Filtered out {len(go_samples) - len(valid_pairs)} failed tasks")
    go_samples_valid = [p[0] for p in valid_pairs]
    task_descriptions_valid = [p[1] for p in valid_pairs]
    print(f"  -> Final task count: {len(task_descriptions_valid)}")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_go_tasks(go_samples_valid, task_descriptions_valid, "stack-go")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_stack-go-v3-test")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(task_descriptions_valid)} Go tasks!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
