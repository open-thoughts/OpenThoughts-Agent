#!/usr/bin/env python3
"""
Generate tasks from Bash verifier scripts in Nemotron.
Filters for bash scripts that ARE test/verifier scripts, uses them as test.sh,
and generates instructions synthetically.
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
from data.commons import create_standard_task_toml, create_standard_dockerfile, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 5_000
MODEL = "gpt-5-nano-2025-08-07"

# =============================================================================
# LLM Prompts
# =============================================================================

VERIFIER_TO_TASK_PROMPT = """You are an expert at creating coding tasks from test/verifier scripts.

Given the following bash test script (a verifier), create a task description for what needs to be implemented to make this verifier pass. The task should:
1. Describe what functionality needs to be built
2. Specify expected inputs and outputs
3. Be clear about file locations and naming (solution should be in /app/)
4. NOT reveal the exact test logic, but give enough info to implement correctly

The verifier script tests:
{{text}}

Create a task description (just the task, no preamble). The solution should be placed in /app/:"""

# =============================================================================
# Verifier Detection Patterns
# =============================================================================

# Patterns that indicate a script is a test/verifier
VERIFIER_PATTERNS = [
    r'assert\s+',                          # assert statements
    r'test\s+-[efdrsxw]',                  # file test operators
    r'\[\s+-[efdrsxw]\s+',                 # [ -e file ] style tests
    r'diff\s+',                            # comparing outputs
    r'cmp\s+',                             # comparing files
    r'exit\s+[01]',                        # explicit exit codes
    r'echo\s+["\']?(PASS|FAIL|OK|ERROR)',  # test result output
    r'(passed|failed|success|failure)',    # test keywords (case insensitive checked separately)
    r'\$\?\s*-?(eq|ne|gt|lt)\s*[01]',      # checking exit codes
    r'if\s+.*;\s*then.*exit',              # conditional exit
    r'expected.*actual|actual.*expected',  # comparison keywords
    r'assertEqual|assertEquals',            # test functions
    r'run_test|test_.*\(\)',               # test function patterns
]

# Patterns that indicate it's NOT a verifier (general utility scripts)
NON_VERIFIER_PATTERNS = [
    r'#!/bin/bash\s*\n\s*#.*install',      # Installation scripts
    r'apt-get\s+install',                   # Package installation
    r'curl.*\|\s*bash',                     # Download and run scripts
    r'systemctl\s+(start|stop|enable)',     # Service management
    r'docker\s+(build|run|push)',           # Docker operations
    r'git\s+(clone|pull|push)',             # Git operations (unless testing git)
    r'ssh\s+',                              # SSH scripts
    r'scp\s+',                              # File transfer
    r'backup|restore',                      # Backup scripts
    r'deploy|release',                      # Deployment scripts
]

# Dangerous patterns to exclude
DANGEROUS_PATTERNS = [
    r'rm\s+-rf\s+/',
    r'rm\s+-rf\s+\*',
    r':\s*\(\s*\)\s*\{',                   # Fork bomb start
    r'>\s*/dev/sd[a-z]',
    r'dd\s+if=.*of=/dev/',
    r'mkfs\.',
]

# Patterns indicating script is NOT self-contained (needs external tools/files)
NON_SELFCONTAINED_PATTERNS = [
    r'/[a-zA-Z0-9_-]+\.(sh|py|pl|rb)',     # Absolute paths to scripts (e.g., /strip-timestamps.sh)
    r'^\s*\./[a-zA-Z0-9_-]+',               # Relative script calls that may not exist
    r'source\s+[^/]',                       # Sourcing external scripts
    r'\.\s+[^/]',                           # Dot-sourcing external scripts
    r'require\s+',                          # Ruby/Perl requires
    r'import\s+',                           # Python imports (shouldn't be in bash)
    r'#include\s+',                         # C includes (shouldn't be in bash)
    r'marian|strip-timestamps|diff-nums',  # Known problematic tools
    r'/usr/local/[a-z]+/[a-z]+',           # Custom tool paths
    r'\.jar\s',                             # Java jars
    r'cargo\s+run',                         # Rust compilation
    r'go\s+run',                            # Go compilation (takes too long)
    r'npm\s+run',                           # Node scripts
    r'make\s+',                             # Makefiles
    r'cmake\s+',                            # CMake
    # System configuration patterns (NOT tests)
    r'/etc/rc\.conf',                       # FreeBSD/Linux system config
    r'/boot/loader\.conf',                  # FreeBSD boot config
    r'sysrc\s+',                            # FreeBSD sysrc command
    r'sysctl\s+hw\.',                       # Hardware sysctl queries
    r'rc-update',                           # OpenRC service management
    r'systemctl\s+',                        # Systemd commands
    r'/usr/home',                           # FreeBSD home directory
    r'pciconf|devinfo',                     # FreeBSD device info
    r'ifconfig\s+-l',                       # Network interface listing
    r'/var/log/',                           # Log directories
    r'apt-get|yum|dnf|pacman|pkg\s+install', # Package managers
    r'chmod\s+\+x\s+/',                     # Making system files executable
    r'chown\s+root',                        # Changing ownership to root
    r'>/dev/null\s+2>&1\s*$',               # Silencing output (not test-like)
]


# =============================================================================
# Filtering Functions
# =============================================================================


def has_shebang(content: str) -> bool:
    """Check if content starts with a bash shebang."""
    first_line = content.strip().split('\n')[0] if content.strip() else ''
    return bool(re.match(r'^#!\s*/(?:usr/)?bin/(?:ba)?sh', first_line))


def count_code_lines(content: str) -> int:
    """Count non-comment, non-empty lines."""
    lines = content.split('\n')
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            code_lines += 1
    return code_lines


def is_dangerous(content: str) -> bool:
    """Check if script contains dangerous patterns."""
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False


def is_self_contained(content: str) -> bool:
    """Check if script is self-contained (doesn't need external tools/files)."""
    for pattern in NON_SELFCONTAINED_PATTERNS:
        if re.search(pattern, content, re.MULTILINE):
            return False

    # Also reject scripts that call many external commands (likely need specific tools)
    # Standard commands that are safe
    safe_commands = {
        'echo', 'printf', 'cat', 'grep', 'awk', 'sed', 'cut', 'sort', 'uniq',
        'head', 'tail', 'wc', 'tr', 'tee', 'xargs', 'find', 'ls', 'cd', 'pwd',
        'mkdir', 'rm', 'cp', 'mv', 'chmod', 'chown', 'touch', 'test', 'true', 'false',
        'diff', 'cmp', 'exit', 'return', 'read', 'sleep', 'date', 'basename', 'dirname',
        'expr', 'seq', 'bc', 'env', 'export', 'set', 'unset', 'shift', 'getopts',
        'if', 'then', 'else', 'fi', 'for', 'do', 'done', 'while', 'until', 'case', 'esac',
        'function', 'local', 'declare', 'typeset', 'trap', 'wait', 'kill', 'jobs', 'bg', 'fg',
        'python', 'python3', 'perl', 'ruby', 'node', 'php', 'bash', 'sh',  # Interpreters ok
    }

    # Extract all command calls (simplified)
    lines = content.split('\n')
    for line in lines:
        # Skip comments
        if line.strip().startswith('#'):
            continue
        # Look for command invocations at start of line or after ; | && ||
        words = re.split(r'[\s;|&]+', line.strip())
        if words and words[0] and not words[0].startswith('$') and not words[0].startswith('['):
            cmd = words[0].split('/')[-1]  # Get basename of command
            # If it looks like a command (not variable assignment, not control flow)
            if '=' not in cmd and cmd and cmd[0].isalpha() and cmd not in safe_commands:
                # Allow a few unknown commands, but not too many
                pass  # For now, just rely on pattern matching

    return True


def is_verifier_script(content: str) -> bool:
    """
    Check if a bash script looks like a test/verifier script.

    A verifier script typically:
    - Runs some code and checks the output
    - Uses assertions or comparisons
    - Exits with 0 (pass) or non-zero (fail)
    - Contains test-related keywords
    """
    # Must have shebang
    if not has_shebang(content):
        return False

    # Check line count (5-150 lines - verifiers shouldn't be too long)
    code_lines = count_code_lines(content)
    if code_lines < 5 or code_lines > 150:
        return False

    # Check for dangerous patterns
    if is_dangerous(content):
        return False

    # Check if script is self-contained
    if not is_self_contained(content):
        return False

    # Check for non-verifier patterns (utility scripts)
    for pattern in NON_VERIFIER_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return False

    # Reject scripts likely to hang (infinite loops, long sleeps, background processes)
    if re.search(r'while\s+true|while\s+:\s*;|while\s+\[\s*1\s*\]', content):
        return False
    if re.search(r'sleep\s+(\d+)', content):
        match = re.search(r'sleep\s+(\d+)', content)
        if match and int(match.group(1)) > 10:
            return False
    # Background process forks that may never terminate
    if re.search(r'[^|]&\s*$', content, re.MULTILINE):
        return False

    # Count verifier patterns found
    verifier_score = 0
    content_lower = content.lower()

    for pattern in VERIFIER_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            verifier_score += 1

    # Also check for common test keywords
    test_keywords = ['test', 'check', 'verify', 'assert', 'expect', 'pass', 'fail']
    for keyword in test_keywords:
        if keyword in content_lower:
            verifier_score += 0.5

    # Need at least 2 verifier indicators
    return verifier_score >= 2


def adapt_verifier_for_harbor(content: str) -> str:
    """
    Adapt a verifier script to work with harbor format.
    Ensures it writes to /logs/verifier/reward.txt.
    """
    # Check if it already writes to reward.txt
    if '/logs/verifier/reward.txt' in content:
        return content

    # Add harbor-compatible wrapper
    # NOTE: Do NOT use set -e at wrapper level - it would exit before writing reward.txt
    adapted = '''#!/bin/bash

# Create logs directory
mkdir -p /logs/verifier

# Ensure reward.txt is always written, even on unexpected errors
trap 'echo "0" > /logs/verifier/reward.txt' ERR

# Run original verifier and capture exit code
run_verifier() {
'''

    # Indent original content (skip shebang and any set -e in original)
    lines = content.split('\n')
    start_idx = 1 if lines[0].startswith('#!') else 0
    for line in lines[start_idx:]:
        adapted += '    ' + line + '\n'

    adapted += '''}

# Execute with timeout (120s) to prevent hangs from infinite loops, sleeps, or network waits
if timeout 120 bash -c "$(declare -f run_verifier); run_verifier"; then
    echo "1" > /logs/verifier/reward.txt
    echo "Tests passed!"
    exit 0
else
    echo "0" > /logs/verifier/reward.txt
    echo "Tests failed!"
    exit 1
fi
'''
    return adapted


def filter_verifiers_from_nemotron(limit: int, max_scan: int = 50_000_000) -> List[Dict]:
    """
    Filter Nemotron for bash verifier/test scripts.

    Args:
        limit: Maximum number of verifier samples to collect
        max_scan: Maximum number of samples to scan before stopping

    Returns:
        List of verifier samples with content and metadata
    """
    print(f"Loading Nemotron (Shell subset, streaming)...")
    ds = load_nemotron_stream("shell")

    verifier_samples = []
    scanned = 0

    print(f"Scanning for verifier scripts (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding verifiers")

    for raw_sample in itertools.islice(ds, max_scan):
        scanned += 1

        # CC-Code-v1: extract bash code blocks from web pages
        if "text" in raw_sample and "content" not in raw_sample:
            content = extract_code_from_sample(raw_sample, "shell")
            if not content:
                continue
            sample = {"content": content, "path": raw_sample.get("uuid", ""), "repo": "cc-code-v1", "size": len(content)}
        else:
            sample = normalize_sample(raw_sample)
            content = sample['content']

        if is_verifier_script(content):
            # Adapt for harbor format
            adapted_content = adapt_verifier_for_harbor(content)

            verifier_samples.append({
                'text': content,  # Original for LLM prompt
                'adapted_text': adapted_content,  # Adapted for test.sh
                'path': sample['path'],
                'repo': sample['repo'],
                'size': sample['size'],
                'code_lines': count_code_lines(content),
            })
            pbar.update(1)

            if len(verifier_samples) >= limit:
                break

        if scanned % 10000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(verifier_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(verifier_samples)} verifier scripts")
    return verifier_samples


# =============================================================================
# Task Generation Functions
# =============================================================================


def create_harbor_task_directory(
    output_dir: Path,
    task_id: int,
    instruction_content: str,
    test_sh_content: str,
    original_script: str,
    dataset_prefix: str,
    metadata: Optional[Dict] = None,
) -> Path:
    """Create a harbor-format task directory."""
    task_dir = output_dir / f"{dataset_prefix}-{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Create environment directory with STANDARD Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(create_standard_dockerfile(), encoding="utf-8")

    # Create tests directory
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    # Write test.sh (the adapted verifier script)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(test_sh_content, encoding="utf-8")
    os.chmod(test_sh_path, 0o755)

    # Create instruction.md (LLM-generated)
    instruction_path = task_dir / "instruction.md"
    instruction_path.write_text(instruction_content, encoding="utf-8")

    # Create task.toml
    task_toml_path = task_dir / "task.toml"
    task_toml_path.write_text(create_standard_task_toml(), encoding="utf-8")

    # Create metadata.json with original script
    if metadata is None:
        metadata = {}
    metadata["original_verifier"] = original_script
    metadata["source_dataset"] = "nemotron"
    metadata_path = task_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_bash_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "nemotron-bash",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction) in enumerate(tqdm(
        zip(samples, instructions),
        total=len(samples),
        desc="Creating task directories"
    )):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
            "source_size": sample.get("size", 0),
            "code_lines": sample.get("code_lines", 0),
        }

        create_harbor_task_directory(
            output_dir=temp_dir,
            task_id=i,
            instruction_content=instruction,
            test_sh_content=sample["adapted_text"],
            original_script=sample["text"],
            dataset_prefix=dataset_prefix,
            metadata=metadata,
        )

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


# =============================================================================
# Main Pipeline
# =============================================================================


def main() -> None:
    """Main pipeline for generating bash verifier tasks."""

    # Step 1: Filter for verifier scripts from Nemotron
    print("Step 1: Filtering verifier scripts from Nemotron...")
    verifier_samples = filter_verifiers_from_nemotron(LIMIT)
    print(f"  -> {len(verifier_samples)} verifier scripts found")

    if not verifier_samples:
        print("\nNo verifier samples found. Exiting.")
        return

    # Step 2: Generate task instructions via LLM
    print("\nStep 2: Generating task instructions from verifiers...")
    dataset = Dataset.from_list(verifier_samples)

    instruction_result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": VERIFIER_TO_TASK_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    instruction_dataset = instruction_result.dataset
    instructions = instruction_dataset["task_description"]
    print(f"  -> Generated {len(instructions)} task descriptions")

    # Step 3: Generate harbor task directories
    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_bash_tasks(verifier_samples, instructions, "nemotron-bash")
    print(f"  -> Task directory: {task_dir}")

    # Step 4: Upload to HuggingFace
    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_nemotron-bash")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated and uploaded {len(verifier_samples)} bash tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
