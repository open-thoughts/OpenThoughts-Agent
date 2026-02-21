#!/usr/bin/env python3
"""
Generate tasks from Dockerfiles in The Stack.

The original Dockerfiles from The Stack are used as INSPIRATION to understand
what development environment is needed. An LLM then generates a fresh, minimal,
buildable Dockerfile using modern base images, along with a task and tests.
"""

import argparse
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

# =============================================================================
# LLM Prompts
# =============================================================================

DOCKERFILE_REWRITE_PROMPT = """You are a Docker expert. Given this old Dockerfile from a public repository, write a NEW minimal Dockerfile that sets up a similar development environment using modern, currently-available base images and packages.

Original Dockerfile (for inspiration only -- do NOT copy it verbatim):
{{text}}

STRICT requirements for your new Dockerfile:
1. Use EXACTLY ONE of these base images (pick the most appropriate):
   - python:3.12-slim  (for Python projects)
   - node:20-slim  (for Node.js/JavaScript projects)
   - golang:1.22  (for Go projects)
   - ruby:3.3-slim  (for Ruby projects)
   - rust:1.77  (for Rust projects)
   - ubuntu:22.04  (for C/C++, Java, multi-language, or general projects)
   - debian:bookworm-slim  (for lightweight general projects)
2. For apt-get packages, ONLY use packages that exist in the default Ubuntu/Debian repos. Do NOT use packages that require adding PPAs or external repos. Safe packages include: build-essential, git, curl, wget, ca-certificates, python3, python3-pip, python3-venv, nodejs, npm, default-jdk, maven, gradle, gcc, g++, make, cmake, pkg-config, vim, nano, jq, zip, unzip, ffmpeg, sqlite3, libsqlite3-dev, postgresql-client, redis-tools, nginx, openssh-client.
   AVOID these packages (they are NOT in default repos): kubectl, terraform, jsonnet, chromaprint, helm, docker, docker-compose, ansible, awscli, gcloud.
3. For pip packages, use: pip install --no-cache-dir --break-system-packages <packages>
4. Do NOT clone git repos, download from URLs, or use wget/curl to fetch binaries
5. Do NOT use COPY or ADD instructions
6. Do NOT use third-party mirrors, PPAs, or custom package repos
7. Do NOT use multi-stage builds (no "FROM ... as ...")
8. Set WORKDIR to /app
9. Keep it VERY minimal -- only install 3-6 packages beyond the base image
10. End with CMD ["bash"]

Output ONLY the Dockerfile content (no markdown fences, no explanation):"""

DOCKERFILE_INSTRUCTION_PROMPT = """You are an expert at creating coding tasks.

Given the following Dockerfile that sets up a development environment, create a coding task that an AI agent could complete IN this environment. The task should:
1. Make use of the tools/languages/frameworks installed in this Dockerfile
2. Be a realistic, self-contained coding task (e.g., "Write a Python script that...", "Create a web server that...")
3. Have clear requirements and expected behavior
4. NOT be about Docker or modifying the Dockerfile itself

The agent's solution should be placed in /app/ directory.

Dockerfile (this is the environment the task will run in):
{{rewritten_dockerfile}}

Create a task description that makes sense for this environment (just the task, no preamble):"""

DOCKERFILE_TEST_PROMPT = """You are an expert at creating test scripts.

Given the following task description and the Dockerfile environment it runs in, create a test.sh script that verifies the solution works correctly.

The test script should:
1. Run the solution (in /app/ directory)
2. Verify it produces correct outputs
3. Write "1" to /logs/verifier/reward.txt if tests pass
4. Write "0" to /logs/verifier/reward.txt if tests fail
5. Start with #!/bin/bash and create /logs/verifier directory

Task description:
{{task_description}}

Dockerfile environment:
{{rewritten_dockerfile}}

Output only the shell script code (no markdown fences):"""

# =============================================================================
# Filtering Functions
# =============================================================================


def calculate_dockerfile_complexity(content: str) -> int:
    """Calculate complexity score for a Dockerfile."""
    score = 0

    # Multi-stage build
    from_count = len(re.findall(r'^FROM\s+', content, re.MULTILINE | re.IGNORECASE))
    if from_count > 1:
        score += 2

    # Package managers
    if re.search(r'apt-get\s+install', content, re.IGNORECASE):
        score += 1
    if re.search(r'pip\s+install|pip3\s+install', content, re.IGNORECASE):
        score += 1
    if re.search(r'npm\s+install', content, re.IGNORECASE):
        score += 1
    if re.search(r'yum\s+install', content, re.IGNORECASE):
        score += 1
    if re.search(r'apk\s+add', content, re.IGNORECASE):
        score += 1

    # Instructions
    if re.search(r'^COPY\s+', content, re.MULTILINE | re.IGNORECASE):
        score += 1
    if re.search(r'^EXPOSE\s+', content, re.MULTILINE | re.IGNORECASE):
        score += 1
    if re.search(r'^ENV\s+', content, re.MULTILINE | re.IGNORECASE):
        score += 1
    if re.search(r'^WORKDIR\s+', content, re.MULTILINE | re.IGNORECASE):
        score += 1

    return score


def extract_environment_info(content: str) -> Dict:
    """Extract useful info about what's installed in the Dockerfile."""
    info = {
        "base_image": "",
        "languages": [],
        "frameworks": [],
        "tools": [],
    }

    # Extract base image
    match = re.search(r'^FROM\s+(\S+)', content, re.MULTILINE | re.IGNORECASE)
    if match:
        info["base_image"] = match.group(1)

    # Detect languages
    if re.search(r'python|pip', content, re.IGNORECASE):
        info["languages"].append("python")
    if re.search(r'node|npm|yarn', content, re.IGNORECASE):
        info["languages"].append("nodejs")
    if re.search(r'ruby|gem|bundler', content, re.IGNORECASE):
        info["languages"].append("ruby")
    if re.search(r'golang|go\s+build', content, re.IGNORECASE):
        info["languages"].append("go")
    if re.search(r'rustc|cargo', content, re.IGNORECASE):
        info["languages"].append("rust")
    if re.search(r'java|maven|gradle', content, re.IGNORECASE):
        info["languages"].append("java")

    # Detect frameworks
    if re.search(r'flask', content, re.IGNORECASE):
        info["frameworks"].append("flask")
    if re.search(r'django', content, re.IGNORECASE):
        info["frameworks"].append("django")
    if re.search(r'express', content, re.IGNORECASE):
        info["frameworks"].append("express")
    if re.search(r'react', content, re.IGNORECASE):
        info["frameworks"].append("react")
    if re.search(r'rails', content, re.IGNORECASE):
        info["frameworks"].append("rails")

    return info


def is_valid_dockerfile(content: str) -> bool:
    """Check if content is a valid Dockerfile suitable for environment inspiration.

    We filter aggressively since the original Dockerfile is only used as
    inspiration -- the LLM will rewrite it. We want Dockerfiles that clearly
    describe a useful development environment.
    """
    # Must have FROM instruction
    if not re.search(r'^FROM\s+', content, re.MULTILINE | re.IGNORECASE):
        return False

    # Count non-comment, non-empty lines
    lines = [l.strip() for l in content.split('\n')
             if l.strip() and not l.strip().startswith('#')]
    if len(lines) < 5:
        return False

    # Should install something useful (not just a bare image)
    has_install = re.search(r'apt-get|pip|npm|yum|apk|gem|cargo', content, re.IGNORECASE)
    if not has_install:
        return False

    # Reject Dockerfiles with COPY or ADD that reference external files
    if re.search(r'^COPY\s+(?!--from)', content, re.MULTILINE | re.IGNORECASE):
        return False
    if re.search(r'^ADD\s+', content, re.MULTILINE | re.IGNORECASE):
        return False

    # ---- Additional filters to reject problematic Dockerfiles ----

    # Reject ARM / non-amd64 base images (balenalib, arm32v7, arm64v8, etc.)
    base_match = re.search(r'^FROM\s+(\S+)', content, re.MULTILINE | re.IGNORECASE)
    if base_match:
        base_image = base_match.group(1).lower()
        arm_patterns = [
            'balenalib/', 'arm32v7/', 'arm64v8/', 'armhf',
            'aarch64', 'ppc64le', 's390x', 'mips',
            'orange-pi', 'beaglebone', 'raspberry',
        ]
        if any(p in base_image for p in arm_patterns):
            return False

        # Reject EOL distros
        eol_patterns = [
            'centos:5', 'centos:6', 'centos:7', 'centos:8',
            'ubuntu:14', 'ubuntu:16', 'ubuntu:18',
            'ubuntu:cosmic', 'ubuntu:disco', 'ubuntu:eoan',
            'ubuntu:hirsute', 'ubuntu:impish',
            'debian:jessie', 'debian:stretch', 'debian:wheezy',
            'fedora:2', 'fedora:3',  # fedora 20-39 — too old
            'alpine:3.1', 'alpine:3.2', 'alpine:3.3', 'alpine:3.4',
            'alpine:3.5', 'alpine:3.6', 'alpine:3.7', 'alpine:3.8',
            'alpine:3.9',
        ]
        if any(base_image.startswith(p) or f'/{p}' in base_image for p in eol_patterns):
            return False

    # Reject third-party mirrors (Chinese mirrors, custom S3 buckets, etc.)
    mirror_patterns = [
        'mirrors.tuna.tsinghua.edu.cn', 'mirrors.aliyun.com',
        'mirrors.ustc.edu.cn', 'registry.npmmirror.com',
        'resin-packages.s3.amazonaws.com',
    ]
    content_lower = content.lower()
    if any(m in content_lower for m in mirror_patterns):
        return False

    # Reject very old language versions in the base image tag
    old_lang_patterns = [
        r'golang:1\.[0-9]\b', r'golang:1\.1[0-7]\b',  # Go < 1.18
        r'node:(?:[0-9]|1[0-5])-',  # Node < 16
        r'python:2\.', r'python:3\.[0-6][^0-9]',  # Python < 3.7
        r'ruby:2\.[0-5]',  # Ruby < 2.6
    ]
    if base_match:
        for pat in old_lang_patterns:
            if re.search(pat, base_match.group(1), re.IGNORECASE):
                return False

    return True


def filter_dockerfiles_from_stack(limit: int, max_scan: int = 500_000, min_complexity: int = 3) -> List[Dict]:
    """Filter The Stack for Dockerfile content."""
    print(f"Loading The Stack (Dockerfile subset, streaming)...")
    ds = load_dataset(
        'bigcode/the-stack',
        data_dir='data/dockerfile',
        split='train',
        streaming=True
    )

    dockerfile_samples = []
    scanned = 0

    print(f"Scanning for Dockerfiles (limit={limit}, max_scan={max_scan}, min_complexity={min_complexity})...")
    pbar = tqdm(total=limit, desc="Finding Dockerfiles")

    for sample in itertools.islice(ds, max_scan):
        scanned += 1
        content = sample.get('content', '')

        if is_valid_dockerfile(content):
            complexity = calculate_dockerfile_complexity(content)
            if complexity >= min_complexity:
                env_info = extract_environment_info(content)
                dockerfile_samples.append({
                    'text': content,
                    'path': sample.get('path', ''),
                    'repo': sample.get('repository_name', ''),
                    'size': sample.get('size', 0),
                    'complexity': complexity,
                    'env_info': env_info,
                })
                pbar.update(1)

                if len(dockerfile_samples) >= limit:
                    break

        if scanned % 10000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(dockerfile_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(dockerfile_samples)} valid Dockerfiles")
    return dockerfile_samples


# =============================================================================
# Task Generation Functions
# =============================================================================


def sanitize_rewritten_dockerfile(dockerfile: str) -> str:
    """Post-process LLM-generated Dockerfile to fix common issues."""
    lines = dockerfile.strip().splitlines()
    cleaned = []
    for line in lines:
        # Strip markdown fences the LLM might include
        stripped = line.strip()
        if stripped in ('```dockerfile', '```Dockerfile', '```', '```docker'):
            continue
        cleaned.append(line)

    result = '\n'.join(cleaned).strip()

    # Ensure it starts with FROM
    if not re.search(r'^FROM\s+', result, re.MULTILINE | re.IGNORECASE):
        return ""

    # Reject multi-stage builds (multiple FROM lines) -- they use COPY --from which we banned
    from_count = len(re.findall(r'^FROM\s+', result, re.MULTILINE | re.IGNORECASE))
    if from_count > 1:
        return ""

    # Reject if it uses COPY or ADD
    if re.search(r'^COPY\s+', result, re.MULTILINE | re.IGNORECASE):
        return ""
    if re.search(r'^ADD\s+', result, re.MULTILINE | re.IGNORECASE):
        return ""

    # Reject if it clones git repos or downloads from URLs
    if re.search(r'git\s+clone', result, re.IGNORECASE):
        return ""
    if re.search(r'(curl|wget)\s+.*http', result, re.IGNORECASE):
        return ""

    # Reject known-bad packages that aren't in default repos
    bad_packages = ['kubectl', 'terraform', 'jsonnet', 'chromaprint', 'helm',
                    'docker-ce', 'docker-compose', 'ansible']
    result_lower = result.lower()
    for pkg in bad_packages:
        if pkg in result_lower:
            return ""

    # Fix common pip issue: add --break-system-packages if missing
    if 'pip install' in result and '--break-system-packages' not in result:
        result = result.replace('pip install', 'pip install --break-system-packages')
    if 'pip3 install' in result and '--break-system-packages' not in result:
        result = result.replace('pip3 install', 'pip3 install --break-system-packages')

    # Ensure WORKDIR /app exists
    if not re.search(r'WORKDIR\s+/app', result, re.IGNORECASE):
        result += '\nWORKDIR /app'

    # Ensure CMD ["bash"] at the end
    if not re.search(r'CMD\s+\[?"bash', result, re.IGNORECASE):
        result += '\nCMD ["bash"]'

    return result


def create_harbor_task_directory(
    output_dir: Path,
    task_id: int,
    instruction_content: str,
    dockerfile_content: str,
    test_code: str,
    dataset_prefix: str,
    metadata: Optional[Dict] = None,
) -> Path:
    """Create a harbor-format task directory."""
    task_dir = output_dir / f"{dataset_prefix}-{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Create environment directory with the REWRITTEN Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")

    # Create tests directory
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    # Write test.sh (LLM-generated)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(test_code, encoding="utf-8")
    os.chmod(test_sh_path, 0o755)

    # Create instruction.md
    instruction_path = task_dir / "instruction.md"
    instruction_path.write_text(instruction_content, encoding="utf-8")

    # Create task.toml
    task_toml_path = task_dir / "task.toml"
    task_toml_path.write_text(create_standard_task_toml(), encoding="utf-8")

    # Create metadata.json
    if metadata is None:
        metadata = {}
    metadata_path = task_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_dockerfile_tasks(
    samples: List[Dict],
    rewritten_dockerfiles: List[str],
    instructions: List[str],
    tests: List[str],
    dataset_prefix: str = "stack-dockerfile",
) -> str:
    """Generate harbor-format task directories using rewritten Dockerfiles."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, dockerfile, instruction, test_code) in enumerate(tqdm(
        zip(samples, rewritten_dockerfiles, instructions, tests),
        total=len(samples),
        desc="Creating task directories"
    )):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
            "source_size": sample.get("size", 0),
            "complexity_score": sample.get("complexity", 0),
            "env_info": sample.get("env_info", {}),
        }

        create_harbor_task_directory(
            output_dir=temp_dir,
            task_id=i,
            instruction_content=instruction,
            dockerfile_content=dockerfile,
            test_code=test_code,
            dataset_prefix=dataset_prefix,
            metadata=metadata,
        )

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


# =============================================================================
# Main Pipeline
# =============================================================================


def main() -> None:
    """Main pipeline for generating Dockerfile-environment tasks."""
    parser = argparse.ArgumentParser(description="Generate dockerfile tasks from The Stack")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Number of tasks to generate")
    args = parser.parse_args()
    limit = args.limit

    # Step 1: Filter Dockerfiles from The Stack
    print("Step 1: Filtering Dockerfiles from The Stack...")
    dockerfile_samples = filter_dockerfiles_from_stack(limit, min_complexity=3)
    print(f"  -> {len(dockerfile_samples)} Dockerfiles found")

    if not dockerfile_samples:
        print("\nNo Dockerfile samples found. Exiting.")
        return

    # Step 2: Rewrite Dockerfiles into fresh, buildable versions via LLM
    print("\nStep 2: Rewriting Dockerfiles with modern base images...")
    dataset = Dataset.from_list(dockerfile_samples)

    rewrite_result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": DOCKERFILE_REWRITE_PROMPT,
            "output_column": "rewritten_dockerfile"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    rewrite_dataset = rewrite_result.dataset

    # Sanitize the rewritten Dockerfiles
    rewritten_raw = rewrite_dataset["rewritten_dockerfile"]
    rewritten_clean = [sanitize_rewritten_dockerfile(d) for d in rewritten_raw]

    # Filter out empty/invalid rewrites
    valid_indices = [i for i, d in enumerate(rewritten_clean) if d.strip()]
    if len(valid_indices) < len(rewritten_clean):
        print(f"  -> Filtered out {len(rewritten_clean) - len(valid_indices)} invalid rewrites")

    # Rebuild dataset with only valid entries
    valid_samples = [dockerfile_samples[i] for i in valid_indices]
    valid_dockerfiles = [rewritten_clean[i] for i in valid_indices]
    valid_records = [dict(rewrite_dataset[i]) for i in valid_indices]
    rewrite_dataset = Dataset.from_list(valid_records)

    print(f"  -> Rewrote {len(valid_dockerfiles)} Dockerfiles successfully")

    # Step 3: Generate task instructions via LLM
    print("\nStep 3: Generating task instructions...")
    instruction_result = run_completions(
        rewrite_dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": DOCKERFILE_INSTRUCTION_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    instruction_dataset = instruction_result.dataset
    print(f"  -> Generated {len(instruction_dataset)} task descriptions")

    # Step 4: Generate test.sh via LLM
    print("\nStep 4: Generating test scripts...")
    test_result = run_completions(
        instruction_dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": DOCKERFILE_TEST_PROMPT,
            "output_column": "test_code"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    final_dataset = test_result.dataset

    instructions = final_dataset["task_description"]
    tests = final_dataset["test_code"]
    print(f"  -> Generated {len(tests)} test files")

    # Step 5: Generate harbor task directories
    print("\nStep 5: Generating harbor task directories...")
    task_dir = generate_dockerfile_tasks(
        valid_samples, valid_dockerfiles, instructions, tests, "stack-dockerfile"
    )
    print(f"  -> Task directory: {task_dir}")

    # Step 6: Upload to HuggingFace
    print("\nStep 6: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_stack-dockerfile")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated and uploaded {len(valid_samples)} tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
