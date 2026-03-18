#!/usr/bin/env python3
"""
Generate tasks from BugSwarm dataset (3,000+ CI environment pairs).

BugSwarm provides reproducible bug-fix pairs with full CI environments.
Each artifact has a failing version (with bug) and passing version (fixed),
extracted from real Travis CI builds.

Languages: Java, Python
Source: https://github.com/BugSwarm
"""

import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))  # Add dclm-mine directory

from dataset_utils import (
    get_cache_dir,
    load_bugswarm_local,
    load_json_file,
)
from data.commons import (
    create_harbor_task_directory_generic,
    create_generic_test_sh,
    create_pytest_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"

# Dockerfile templates for BugSwarm (language-specific)
BUGSWARM_DOCKERFILES = {
    "python": """FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \\
    pytest \\
    pytest-timeout \\
    coverage \\
    tox

RUN mkdir -p /logs/verifier /workspace
""",
    "java": """FROM openjdk:11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    git \\
    maven \\
    gradle \\
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /logs/verifier /workspace
""",
}


def get_bugswarm_dockerfile(language: str) -> str:
    """Get Dockerfile for BugSwarm language."""
    language = language.lower()
    if language in BUGSWARM_DOCKERFILES:
        return BUGSWARM_DOCKERFILES[language]
    return BUGSWARM_DOCKERFILES["python"]


def create_bugswarm_test_sh(language: str, test_cmd: str) -> str:
    """Create test.sh for BugSwarm tasks."""
    return f'''#!/bin/bash
set -e

mkdir -p /logs/verifier

cleanup() {{
    if [ $? -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
    else
        echo "0" > /logs/verifier/reward.txt
    fi
}}
trap cleanup EXIT

cd /workspace

echo "Running BugSwarm tests ({language})..."
{test_cmd} 2>&1 | tee /logs/verifier/test_output.txt

exit ${{PIPESTATUS[0]}}
'''


def load_bugswarm(language: str = None, limit: int = LIMIT, local_json_path: Path = None) -> List[Dict]:
    """
    Load BugSwarm dataset.

    BugSwarm artifacts contain failing/passing Docker image pairs.

    Args:
        language: Filter by language (optional)
        limit: Maximum samples to load
        local_json_path: Path to locally downloaded BugSwarm JSON file.
                         Download from https://www.bugswarm.org/dataset/ by clicking "Export"
    """
    print(f"Loading BugSwarm dataset...")

    # Option 1: Load from local JSON file if provided
    if local_json_path and Path(local_json_path).exists():
        print(f"Loading from local file: {local_json_path}")
        samples = load_bugswarm_local(Path(local_json_path), limit=limit)
        if language:
            samples = [s for s in samples if s.get("language", "").lower() == language.lower()]
        return samples

    # Option 2: Check cache directory for previously downloaded file
    cache_dir = get_cache_dir()
    cached_json = cache_dir / "bugswarm" / "artifacts.json"
    if cached_json.exists():
        print(f"Loading from cached file: {cached_json}")
        samples = load_bugswarm_local(cached_json, limit=limit)
        if language:
            samples = [s for s in samples if s.get("language", "").lower() == language.lower()]
        return samples

    # Option 3: Try HuggingFace sources
    repo_names = [
        "bugswarm/bugswarm",
        "BugSwarm/bugswarm-artifacts",
        "bug-repair/bugswarm",
    ]

    ds = None
    for repo_name in repo_names:
        try:
            ds = load_dataset(repo_name, split="test", streaming=True)
            print(f"Loaded from {repo_name}")
            break
        except Exception as e:
            print(f"Could not load from {repo_name}: {e}")
            continue

    if ds is None:
        # Create sample data for demonstration
        print("BugSwarm dataset not found on HuggingFace.")
        print("")
        print("To load the full BugSwarm dataset:")
        print("  1. Go to https://www.bugswarm.org/dataset/")
        print("  2. Apply filters if desired")
        print("  3. Click 'Export' to download artifacts.json")
        print("  4. Run with --local-json /path/to/artifacts.json")
        print("")
        print("Creating sample data for demonstration...")
        return _create_sample_bugswarm_data(limit)

    samples = []
    print(f"Collecting up to {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        artifact_id = sample.get("artifact_id", sample.get("id", ""))
        repo = sample.get("repo", sample.get("repository", ""))
        lang = sample.get("language", sample.get("lang", "python"))

        if language and lang.lower() != language.lower():
            continue

        if not artifact_id or not repo:
            continue

        samples.append({
            "artifact_id": artifact_id,
            "repo": repo,
            "language": lang,
            "failed_job_id": sample.get("failed_job_id", ""),
            "passed_job_id": sample.get("passed_job_id", ""),
            "failing_code": sample.get("failing_code", sample.get("buggy_code", "")),
            "passing_code": sample.get("passing_code", sample.get("fixed_code", "")),
            "test_code": sample.get("test_code", ""),
            "diff": sample.get("diff", sample.get("patch", "")),
            "failed_commit": sample.get("failed_commit", ""),
            "passed_commit": sample.get("passed_commit", ""),
            "build_system": sample.get("build_system", ""),
            "test_framework": sample.get("test_framework", ""),
        })

    print(f"Loaded {len(samples)} BugSwarm artifacts")
    return samples


def _create_sample_bugswarm_data(limit: int) -> List[Dict]:
    """Create sample BugSwarm data for demonstration."""
    samples = [
        {
            "artifact_id": "alibaba-fastjson-123456",
            "repo": "alibaba/fastjson",
            "language": "java",
            "failing_code": '''public class JSONParser {
    public Object parse(String json) {
        // Bug: Doesn't handle empty strings
        if (json.startsWith("{")) {
            return parseObject(json);
        }
        return parseArray(json);
    }

    private Object parseObject(String json) {
        // Implementation
        return new JSONObject();
    }

    private Object parseArray(String json) {
        return new JSONArray();
    }
}''',
            "passing_code": '''public class JSONParser {
    public Object parse(String json) {
        if (json == null || json.isEmpty()) {
            return null;
        }
        if (json.startsWith("{")) {
            return parseObject(json);
        }
        return parseArray(json);
    }

    private Object parseObject(String json) {
        return new JSONObject();
    }

    private Object parseArray(String json) {
        return new JSONArray();
    }
}''',
            "test_code": '''import org.junit.Test;
import static org.junit.Assert.*;

public class JSONParserTest {
    @Test
    public void testParseObject() {
        JSONParser parser = new JSONParser();
        assertNotNull(parser.parse("{}"));
    }

    @Test
    public void testParseEmptyString() {
        JSONParser parser = new JSONParser();
        assertNull(parser.parse(""));
    }

    @Test
    public void testParseNull() {
        JSONParser parser = new JSONParser();
        assertNull(parser.parse(null));
    }
}''',
            "diff": '''--- a/JSONParser.java
+++ b/JSONParser.java
@@ -1,6 +1,9 @@
 public class JSONParser {
     public Object parse(String json) {
+        if (json == null || json.isEmpty()) {
+            return null;
+        }
         if (json.startsWith("{")) {
''',
            "build_system": "maven",
            "test_framework": "junit",
        },
        {
            "artifact_id": "requests-requests-789012",
            "repo": "psf/requests",
            "language": "python",
            "failing_code": '''def get(url, params=None, **kwargs):
    """Sends a GET request."""
    # Bug: Doesn't validate URL
    kwargs.setdefault('allow_redirects', True)
    return request('get', url, params=params, **kwargs)

def post(url, data=None, json=None, **kwargs):
    """Sends a POST request."""
    return request('post', url, data=data, json=json, **kwargs)
''',
            "passing_code": '''def get(url, params=None, **kwargs):
    """Sends a GET request."""
    if not url:
        raise ValueError("URL is required")
    kwargs.setdefault('allow_redirects', True)
    return request('get', url, params=params, **kwargs)

def post(url, data=None, json=None, **kwargs):
    """Sends a POST request."""
    if not url:
        raise ValueError("URL is required")
    return request('post', url, data=data, json=json, **kwargs)
''',
            "test_code": '''import pytest
from requests import get, post

def test_get_with_url():
    # Should work with valid URL
    response = get("https://httpbin.org/get")
    assert response is not None

def test_get_without_url():
    with pytest.raises(ValueError):
        get("")

def test_post_without_url():
    with pytest.raises(ValueError):
        post("")
''',
            "diff": '''--- a/requests/api.py
+++ b/requests/api.py
@@ -1,5 +1,7 @@
 def get(url, params=None, **kwargs):
     """Sends a GET request."""
+    if not url:
+        raise ValueError("URL is required")
     kwargs.setdefault('allow_redirects', True)
''',
            "build_system": "pip",
            "test_framework": "pytest",
        },
    ]

    return samples[:limit]


def create_instruction_from_bugswarm(sample: Dict) -> str:
    """Create task instruction from BugSwarm sample."""
    artifact_id = sample.get("artifact_id", "Unknown")
    repo = sample.get("repo", "Unknown")
    language = sample.get("language", "Unknown")
    failing_code = sample.get("failing_code", "")
    build_system = sample.get("build_system", "")
    test_framework = sample.get("test_framework", "")

    instruction = f"""# CI Bug Fix Task (BugSwarm)

## Repository
{repo}

## Artifact ID
{artifact_id}

## Language
{language.capitalize()}

## Build System
{build_system if build_system else "Standard"}

## Test Framework
{test_framework if test_framework else "Standard"}

## Failing Code

The following code causes the CI build to fail:

```{language}
{failing_code}
```

## Your Task

This code was extracted from a real CI failure. Your task is to:

1. Analyze the failing code to identify the bug
2. Fix the bug so that all tests pass
3. Ensure your fix is minimal and targeted
4. Place your fixed code in `/workspace/`

## Context

This is a real bug-fix pair extracted from a Travis CI build:
- The failing version caused CI tests to fail
- The passing version fixed the issue
- Your goal is to reproduce the fix

## Notes

- The fix should be minimal - only change what's necessary
- Consider edge cases that might cause test failures
- The test framework will run the same tests that failed in CI
"""

    return instruction


def get_test_filename(language: str) -> str:
    """Get test filename for language."""
    if language.lower() == "java":
        return "TestSolution.java"
    return "test_solution.py"


def get_test_command(language: str, test_framework: str) -> str:
    """Get test command for language/framework."""
    if language.lower() == "java":
        if test_framework == "gradle":
            return "gradle test"
        return "mvn test -Dtest=TestSolution"
    return "pytest /tests/test_solution.py -v --tb=short"


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a BugSwarm sample."""
    language = sample.get("language", "python")
    instruction = create_instruction_from_bugswarm(sample)

    test_filename = get_test_filename(language)
    test_files = {
        test_filename: sample.get("test_code", ""),
    }

    # Add diff if available
    if sample.get("diff"):
        test_files["expected_fix.patch"] = sample["diff"]

    metadata = {
        "source": "bugswarm",
        "artifact_id": sample.get("artifact_id", ""),
        "repo": sample.get("repo", ""),
        "language": language,
        "failed_job_id": sample.get("failed_job_id", ""),
        "passed_job_id": sample.get("passed_job_id", ""),
        "build_system": sample.get("build_system", ""),
        "test_framework": sample.get("test_framework", ""),
    }

    # Store passing code as solution
    solution_files = None
    if sample.get("passing_code"):
        ext = "java" if language.lower() == "java" else "py"
        solution_files = {f"solution.{ext}": sample["passing_code"]}

    test_framework = sample.get("test_framework", "")
    test_cmd = get_test_command(language, test_framework)

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_bugswarm_dockerfile(language),
        test_sh=create_bugswarm_test_sh(language, test_cmd),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(samples: List[Dict], dataset_prefix: str = "bugswarm") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, sample in enumerate(tqdm(samples, desc="Creating tasks")):
        create_harbor_task(temp_dir, i, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(language: str = None, limit: int = LIMIT, upload: bool = True, local_json_path: Path = None) -> None:
    """Main pipeline for generating BugSwarm tasks."""

    print(f"Step 1: Loading BugSwarm dataset...")
    samples = load_bugswarm(language=language, limit=limit, local_json_path=local_json_path)
    print(f"  -> {len(samples)} artifacts loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating harbor task directories...")
    dataset_prefix = f"bugswarm-{language}" if language else "bugswarm"
    task_dir = generate_tasks(samples, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 3: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} BugSwarm tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from BugSwarm dataset")
    parser.add_argument("--language", choices=["python", "java"],
                        help="Filter by language")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")
    parser.add_argument("--local-json", type=Path,
                        help="Path to locally downloaded BugSwarm JSON file (from bugswarm.org/dataset)")

    args = parser.parse_args()
    main(language=args.language, limit=args.limit, upload=not args.no_upload, local_json_path=args.local_json)
