#!/usr/bin/env python3
"""
Generate tasks from GitHub Actions Workflows dataset (160K workflows).

The dataset provides:
- 160K+ workflow file commit histories
- 32K+ public GitHub repositories
- Test automation workflows (pytest, jest, go test, etc.)
"""

import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))  # Add dclm-mine directory

from dataset_utils import (
    download_ghactions,
    download_from_zenodo,
    extract_archive,
    get_cache_dir,
    load_ghactions_from_zenodo,
)
from data.completions import run_completions
from data.commons import (
    TEST_TO_INSTRUCTION_PROMPT,
    create_harbor_task_directory_generic,
    create_pytest_test_sh,
    create_generic_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 1000
MODEL = "gpt-4o-mini"

# =============================================================================
# Prompts
# =============================================================================

# Prompt to generate executable test code from a GitHub Actions workflow YAML.
# Used in skip-clone mode where we only have the workflow file, not actual source.
WORKFLOW_TO_TEST_CODE_PROMPT = """You are an expert at writing test code. Given a GitHub Actions CI workflow YAML file, generate a **self-contained test file** that validates the project described by the workflow.

Analyze the workflow to understand:
- What language/framework the project uses
- What the project does (from job names, step names, scripts)
- What dependencies are involved

Then write a test file in the **detected language** that:
1. Tests the core functionality implied by the workflow
2. Uses the standard test framework for that language (pytest for Python, Jest for JavaScript, go test for Go, JUnit for Java)
3. Imports from `/app/` where the agent will place their solution
4. Is fully self-contained (no external fixtures or data files)
5. Has at least 3 meaningful test cases

IMPORTANT for Python tests:
- Start with: import sys; sys.path.insert(0, '/app')
- Use pytest style (def test_xxx)

IMPORTANT for JavaScript tests:
- Use require() NOT import (no ES6 modules)
- Use: const {{ ... }} = require('/app/solution') or similar
- Use describe/it/expect from Jest

Return ONLY the test code, no markdown fences, no explanation.

Workflow YAML:
{{text}}

Test code:"""

# Prompt to generate task instructions from workflow YAML (not raw test code).
WORKFLOW_TO_INSTRUCTION_PROMPT = """You are an expert at creating coding tasks from CI/CD workflow configurations.

Given the following GitHub Actions workflow YAML, create a clear, specific task description that an AI agent could complete. Analyze the workflow to understand what the project does, then describe what needs to be built.

The task should:
1. Describe what functionality needs to be implemented based on what the CI workflow tests/builds
2. Include specific requirements inferred from the workflow steps (language, framework, dependencies)
3. Be self-contained and actionable
4. The solution should be placed in /app/

Workflow YAML:
{{text}}

Create a task description (just the task, no preamble):"""

# Patterns that indicate test workflows
TEST_PATTERNS = [
    r"pytest", r"npm test", r"yarn test", r"go test",
    r"mvn test", r"gradle test", r"cargo test",
    r"rspec", r"phpunit", r"jest",
]


def is_test_workflow(workflow_content: str) -> bool:
    """Check if a workflow file runs tests."""
    content_lower = workflow_content.lower()
    return any(re.search(pattern, content_lower) for pattern in TEST_PATTERNS)


def extract_test_command(workflow_content: str) -> Optional[str]:
    """Extract the test command from a workflow file."""
    content_lower = workflow_content.lower()

    for pattern in TEST_PATTERNS:
        match = re.search(rf"run:\s*.*({pattern}[^\n]*)", content_lower)
        if match:
            return match.group(1).strip()

    return None


def load_ghactions_with_filter(data_dir: Path, limit: int, filter_func) -> List[Dict]:
    """
    Efficiently load workflow files from a directory, filtering during loading.

    This is more efficient than loading all files then filtering because we can
    stop as soon as we have enough matching samples.
    """
    data_dir = Path(data_dir)
    samples = []
    scanned = 0

    print(f"Scanning workflows from: {data_dir}")

    for workflow_file in data_dir.iterdir():
        if len(samples) >= limit:
            break

        if not workflow_file.is_file():
            continue

        scanned += 1
        if scanned % 10000 == 0:
            print(f"  Scanned {scanned} files, found {len(samples)} test workflows...")

        try:
            content = workflow_file.read_text(encoding='utf-8', errors='ignore')

            # Check if it looks like a valid workflow file
            if not ('name:' in content or 'on:' in content or 'jobs:' in content):
                continue

            # Apply the filter function (e.g., is_test_workflow)
            if not filter_func(content):
                continue

            # Detect language from workflow
            language = "python"  # default
            content_lower = content.lower()
            if "node" in content_lower or "npm" in content_lower:
                language = "javascript"
            elif "go " in content_lower or "go test" in content_lower:
                language = "go"
            elif "mvn" in content_lower or "java" in content_lower:
                language = "java"
            elif "cargo" in content_lower or "rust" in content_lower:
                language = "rust"

            samples.append({
                "workflow_content": content,
                "workflow_path": workflow_file.name,
                "repo_name": workflow_file.name,
                "language": language,
                "test_command": extract_test_command(content),
            })

        except Exception:
            continue

    print(f"  Scanned {scanned} files total, found {len(samples)} test workflows")
    return samples


def load_ghactions(limit: int = LIMIT, local_data_dir: Path = None) -> List[Dict]:
    """
    Load GitHub Actions Workflows dataset from Zenodo/HuggingFace.

    Dataset available at: https://zenodo.org/records/10566003

    Args:
        limit: Maximum samples to load
        local_data_dir: Path to locally downloaded/extracted Zenodo data
    """
    print("Loading GitHub Actions Workflows dataset...")

    # Option 1: Load from local data directory if provided
    if local_data_dir and Path(local_data_dir).exists():
        print(f"Loading from local directory: {local_data_dir}")
        try:
            samples = load_ghactions_with_filter(Path(local_data_dir), limit, is_test_workflow)
            return samples[:limit]
        except Exception as e:
            print(f"Error loading from local directory: {e}")

    # Option 2: Check cache for previously downloaded data
    cache_dir = get_cache_dir()
    # Try multiple possible cache locations (tar extracts with nested structure)
    possible_cache_dirs = [
        cache_dir / "zenodo_10566003" / "workflows_extracted" / "workflows",
        cache_dir / "zenodo_10566003" / "workflows" / "workflows",
        cache_dir / "zenodo_10566003" / "workflows",
    ]

    for cached_dir in possible_cache_dirs:
        if cached_dir.exists() and any(cached_dir.iterdir()):
            # Check if this directory contains workflow files (not just subdirectories)
            first_item = next(cached_dir.iterdir(), None)
            if first_item and first_item.is_file():
                print(f"Loading from cached directory: {cached_dir}")
                try:
                    samples = load_ghactions_with_filter(cached_dir, limit, is_test_workflow)
                    return samples[:limit]
                except Exception as e:
                    print(f"Error loading from cache: {e}")

    # Option 3: Try HuggingFace sources
    repo_names = [
        "github-actions/workflows",
        "ci-datasets/ghactions",
        "code-datasets/github-workflows",
    ]

    ds = None
    for repo_name in repo_names:
        try:
            ds = load_dataset(repo_name, split="train", streaming=True)
            print(f"Loaded from {repo_name}")
            break
        except Exception as e:
            print(f"Could not load from {repo_name}: {e}")
            continue

    if ds is None:
        # Option 4: Try to download from Zenodo
        print("")
        print("HuggingFace sources unavailable. Attempting to download from Zenodo...")
        print("Source: https://zenodo.org/records/10566003")
        try:
            data_dir = download_ghactions()
            samples = load_ghactions_from_zenodo(data_dir, limit=limit * 10)
            samples = [s for s in samples if is_test_workflow(s.get("workflow_content", ""))]
            return samples[:limit]
        except Exception as e:
            print(f"Zenodo download failed: {e}")
            print("")
            print("To load the full GitHub Actions dataset:")
            print("  1. Download workflows.tar.gz from https://zenodo.org/records/10566003")
            print("  2. Extract the archive")
            print("  3. Run with --local-data /path/to/workflows")
            raise ValueError("Could not load GitHub Actions dataset")

    samples = []
    print(f"Collecting {limit} test workflow samples...")

    for sample in tqdm(itertools.islice(ds, limit * 10), desc="Loading samples"):
        workflow_content = sample.get("content", sample.get("workflow", ""))

        if not workflow_content or not is_test_workflow(workflow_content):
            continue

        # Extract repository info
        repo_url = sample.get("repo_url", sample.get("repository_url", ""))
        repo_name = sample.get("repo_name", sample.get("repository", ""))

        # Detect language from workflow
        language = "python"  # default
        if "node" in workflow_content.lower() or "npm" in workflow_content.lower():
            language = "javascript"
        elif "go " in workflow_content.lower() or "go test" in workflow_content.lower():
            language = "go"
        elif "mvn" in workflow_content.lower() or "java" in workflow_content.lower():
            language = "java"
        elif "cargo" in workflow_content.lower() or "rust" in workflow_content.lower():
            language = "rust"

        samples.append({
            "workflow_content": workflow_content,
            "repo_url": repo_url,
            "repo_name": repo_name,
            "language": language,
            "test_command": extract_test_command(workflow_content),
            "workflow_path": sample.get("path", sample.get("file_path", "")),
        })

        if len(samples) >= limit:
            break

    print(f"Loaded {len(samples)} test workflows")
    return samples


def clone_and_extract_tests(repo_url: str, max_files: int = 5) -> Dict:
    """Clone a repository and extract test files."""
    result = {
        "test_files": {},
        "error": None,
    }

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Clone repository (shallow)
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, tmpdir],
                capture_output=True,
                timeout=60
            )

            repo_path = Path(tmpdir)

            # Find test files
            test_patterns = [
                "**/test_*.py", "**/tests/*.py", "**/*_test.py",
                "**/test/*.py", "**/*Test.java", "**/tests/*.js",
                "**/*_test.go", "**/tests/*.go",
            ]

            test_files = []
            for pattern in test_patterns:
                test_files.extend(repo_path.glob(pattern))

            # Extract test file contents
            for tf in test_files[:max_files]:
                try:
                    content = tf.read_text(encoding="utf-8", errors="ignore")
                    if len(content) > 100:
                        result["test_files"][tf.name] = content
                except:
                    pass

    except Exception as e:
        result["error"] = str(e)

    return result


def process_workflows_for_tasks(samples: List[Dict], max_per_repo: int = 3) -> List[Dict]:
    """Process workflow repositories to extract test files."""
    print("Extracting test files from repositories...")
    task_samples = []

    for sample in tqdm(samples[:50], desc="Processing repos"):
        if not sample.get("repo_url"):
            continue

        extracted = clone_and_extract_tests(sample["repo_url"], max_files=max_per_repo)

        if extracted.get("error") or not extracted.get("test_files"):
            continue

        for filename, content in extracted["test_files"].items():
            task_samples.append({
                "text": content,
                "filename": filename,
                "language": sample["language"],
                "repo": sample["repo_name"],
                "repo_url": sample["repo_url"],
                "test_command": sample.get("test_command", ""),
            })

    print(f"Extracted {len(task_samples)} test files")
    return task_samples


def _create_jest_test_sh() -> str:
    """Create a test.sh for Jest that correctly copies test files into /app/tests/
    so Jest's testMatch patterns can discover them."""
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

# Create jest.config.js at runtime to ensure correct paths
cat > /app/jest.config.js << 'EOF'
module.exports = {
  testEnvironment: "node",
  rootDir: "/app",
  testMatch: ["**/tests/**/*.js", "**/*.test.js"],
  moduleDirectories: ["node_modules", "/usr/local/lib/node_modules", "/app", "/app/lib"]
};
EOF

# Install dependencies if package.json exists and has dependencies
if [ -f package.json ] && grep -q '"dependencies"' package.json 2>/dev/null; then
    npm install --silent 2>/dev/null || true
fi

# Auto-detect and install missing npm modules from test file
echo "Checking for missing dependencies..."
REQUIRES=$(grep -oP "require\\(['\"]\\K[^'\"@./][^'\"]*" /tests/test_solution.js 2>/dev/null | sort -u || true)
for pkg in $REQUIRES; do
    # Skip Node.js built-in modules
    case "$pkg" in
        fs|path|util|crypto|http|https|url|querystring|os|child_process|stream|events|buffer|string_decoder|timers|net|assert|module|vm|tty|readline|cluster|dns|dgram|zlib|tls) continue ;;
    esac
    if ! node -e "require('$pkg')" 2>/dev/null; then
        echo "Installing $pkg..."
        npm install --silent "$pkg" 2>/dev/null || true
    fi
done

# Copy test file into /app/tests/ so Jest testMatch can find it
mkdir -p /app/tests
cp /tests/test_solution.js /app/tests/

echo "Running tests..."
timeout 300 npx jest /app/tests/test_solution.js --config /app/jest.config.js --no-coverage 2>&1 | tee /logs/verifier/test_output.txt

exit ${PIPESTATUS[0]}
'''


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory from extracted test file."""
    language = sample.get("language", "python")

    lang_config = {
        "python": ("python", create_pytest_test_sh("/tests/test_solution.py"), "test_solution.py"),
        "javascript": ("node", _create_jest_test_sh(), "test_solution.js"),
        "go": ("go", create_generic_test_sh("go test -v /tests/..."), "solution_test.go"),
        "java": ("java", create_generic_test_sh("mvn test -q"), "TestSolution.java"),
        "rust": ("rust", create_generic_test_sh("cargo test"), "tests.rs"),
    }

    dockerfile_lang, test_sh, test_filename = lang_config.get(
        language, lang_config["python"]
    )

    dockerfile = get_dockerfile(dockerfile_lang)

    # Process test content
    test_content = sample.get("text", "")
    if language == "python" and "sys.path" not in test_content:
        test_content = "import sys\nsys.path.insert(0, '/app')\n\n" + test_content

    test_files = {test_filename: test_content}

    metadata = {
        "source": "ghactions",
        "repo": sample.get("repo", ""),
        "repo_url": sample.get("repo_url", ""),
        "original_filename": sample.get("filename", ""),
        "test_command": sample.get("test_command", ""),
    }

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=test_sh,
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "ghactions",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction) in enumerate(tqdm(
        zip(samples, instructions),
        total=len(samples),
        desc="Creating tasks"
    )):
        create_harbor_task(temp_dir, i, instruction, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(limit: int = LIMIT, skip_clone: bool = False, local_data_dir: Path = None) -> None:
    """Main pipeline for generating GitHub Actions tasks."""

    print("Step 1: Loading GitHub Actions Workflows dataset...")
    samples = load_ghactions(limit=limit, local_data_dir=local_data_dir)
    print(f"  -> {len(samples)} test workflows loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    if not skip_clone:
        print("\nStep 2: Extracting test files from repositories...")
        task_samples = process_workflows_for_tasks(samples)

        if not task_samples:
            print("\nNo task samples extracted. Exiting.")
            return

        print(f"  -> {len(task_samples)} test files extracted")

        # In clone mode, the "text" field already contains real test code.
        # Generate instructions from that test code.
        print("\nStep 3: Generating task descriptions...")
        dataset = Dataset.from_list(task_samples)
        result = run_completions(
            dataset,
            model=MODEL,
            map_type="chat",
            map_config={
                "user_message": TEST_TO_INSTRUCTION_PROMPT,
                "output_column": "task_description"
            },
            max_requests_per_minute=500,
            max_tokens_per_minute=1_000_000,
        )
        instructions = result.dataset["task_description"]
        print(f"  -> Generated {len(instructions)} task descriptions")
    else:
        # In skip-clone mode, we only have workflow YAML -- NOT actual test code.
        # We need TWO LLM passes:
        #   (a) Generate executable test code from the workflow YAML.
        #   (b) Generate task instructions from the workflow YAML.
        print("\nStep 2: Generating test code from workflow YAML (skip-clone mode)...")
        workflow_samples = []
        for sample in samples:
            workflow_samples.append({
                "text": sample.get("workflow_content", ""),
                "filename": sample.get("workflow_path", "workflow.yml"),
                "language": sample.get("language", "python"),
                "repo": sample.get("repo_name", ""),
                "repo_url": sample.get("repo_url", ""),
                "test_command": sample.get("test_command", ""),
            })

        dataset = Dataset.from_list(workflow_samples)

        # (a) Generate real test code from workflow YAML
        print("  Generating test code from workflows...")
        test_result = run_completions(
            dataset,
            model=MODEL,
            map_type="chat",
            map_config={
                "user_message": WORKFLOW_TO_TEST_CODE_PROMPT,
                "output_column": "generated_test_code"
            },
            max_requests_per_minute=500,
            max_tokens_per_minute=1_000_000,
        )
        # Extract generated test code from result (handle different column names)
        test_ds = test_result.dataset if hasattr(test_result, 'dataset') else test_result
        test_col = "generated_test_code"
        if test_col not in test_ds.column_names:
            # Curator may use a different column name; find the new one
            new_cols = [c for c in test_ds.column_names if c not in dataset.column_names]
            if new_cols:
                test_col = new_cols[0]
                print(f"  (Note: output column was '{test_col}' instead of 'generated_test_code')")
            else:
                raise ValueError(f"No new columns found in result. Columns: {test_ds.column_names}")
        generated_tests = test_ds[test_col]
        print(f"  -> Generated {len(generated_tests)} test files")

        # (b) Generate task instructions from workflow YAML
        print("  Generating task descriptions from workflows...")
        instr_result = run_completions(
            dataset,
            model=MODEL,
            map_type="chat",
            map_config={
                "user_message": WORKFLOW_TO_INSTRUCTION_PROMPT,
                "output_column": "task_description"
            },
            max_requests_per_minute=500,
            max_tokens_per_minute=1_000_000,
        )
        instr_ds = instr_result.dataset if hasattr(instr_result, 'dataset') else instr_result
        instr_col = "task_description"
        if instr_col not in instr_ds.column_names:
            new_cols = [c for c in instr_ds.column_names if c not in dataset.column_names]
            if new_cols:
                instr_col = new_cols[0]
        instructions = instr_ds[instr_col]
        print(f"  -> Generated {len(instructions)} task descriptions")

        # Build task_samples with the LLM-generated test code (not the YAML)
        task_samples = []
        for ws, test_code in zip(workflow_samples, generated_tests):
            # Strip markdown fences if the LLM wrapped the code in them
            cleaned = _strip_markdown_fences(test_code)
            task_samples.append({
                "text": cleaned,
                "filename": ws["filename"],
                "language": ws["language"],
                "repo": ws["repo"],
                "repo_url": ws["repo_url"],
                "test_command": ws["test_command"],
            })

    print(f"\nStep {'3' if skip_clone else '4'}: Generating harbor task directories...")
    task_dir = generate_tasks(task_samples, instructions, "ghactions")
    print(f"  -> Task directory: {task_dir}")

    print(f"\nStep {'4' if skip_clone else '5'}: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_ghactions")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(task_samples)} GitHub Actions tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences (```python ... ```) if present."""
    code = code.strip()
    # Match opening fence with optional language tag
    if code.startswith("```"):
        # Remove first line (```python, ```javascript, ```, etc.)
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline + 1:]
    # Remove closing fence
    if code.endswith("```"):
        code = code[:-3].rstrip()
    return code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from GitHub Actions dataset")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum workflows to process")
    parser.add_argument("--skip-clone", action="store_true", help="Skip repository cloning")
    parser.add_argument("--local-data", type=Path,
                        help="Path to locally downloaded Zenodo data (from zenodo.org/records/10566003)")

    args = parser.parse_args()
    main(limit=args.limit, skip_clone=args.skip_clone, local_data_dir=args.local_data)
