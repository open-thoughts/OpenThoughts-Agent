#!/usr/bin/env python3
"""
Quick test: Generate small datasets (50 tasks) to verify environment fixes.
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

sys.path.append(str(Path(__file__).parent.parent))

from data.completions import run_completions
from data.commons import create_standard_task_toml, create_standard_dockerfile, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 25  # Small limit for testing
MODEL = "gpt-5-nano-2025-08-07"
TEST_SUFFIX = "-v2-test"

# =============================================================================
# Go Generator with bash fix
# =============================================================================

def filter_go_tests(limit: int) -> List[Dict]:
    ds = load_dataset('bigcode/the-stack', data_dir='data/go', split='train', streaming=True)
    samples = []
    for sample in itertools.islice(ds, 500_000):
        content = sample.get('content', '')
        if '"testing"' in content and 'func Test' in content:
            if len(re.findall(r'func\s+Test\w+\s*\(', content)) >= 2:
                samples.append({'text': content, 'path': sample.get('path', ''), 'repo': sample.get('repository_name', '')})
                if len(samples) >= limit:
                    break
    return samples

def create_go_dockerfile():
    return """FROM golang:1.22-alpine

WORKDIR /app

# Install bash (required by Harbor agent), git, build tools for cgo, and util-linux (for script command)
RUN apk add --no-cache bash git gcc musl-dev util-linux
"""

def create_go_test_sh():
    return '''#!/bin/bash
set -e
mkdir -p /logs/verifier
cleanup() { if [ $? -eq 0 ]; then echo "1" > /logs/verifier/reward.txt; else echo "0" > /logs/verifier/reward.txt; fi; }
trap cleanup EXIT
cd /app
if [ ! -f go.mod ]; then go mod init solution; fi
cp /tests/solution_test.go .
# Resolve dependencies
go mod tidy 2>&1 || true
go mod download 2>&1 || true
go test -v ./... 2>&1 | tee /logs/verifier/test_output.txt
exit ${PIPESTATUS[0]}
'''

# =============================================================================
# Pytest Generator with venv fix
# =============================================================================

def filter_pytest(limit: int) -> List[Dict]:
    ds = load_dataset('bigcode/the-stack', data_dir='data/python', split='train', streaming=True)
    # Allowlist of safe, pip-installable packages
    safe_packages = {
        'pytest', 'os', 'sys', 're', 'json', 'math', 'collections', 'itertools',
        'functools', 'typing', 'unittest', 'pathlib', 'datetime', 'time', 'copy',
        'io', 'string', 'random', 'abc', 'contextlib', 'warnings', 'operator',
        'dataclasses', 'enum', 'textwrap', 'inspect', 'types', 'numbers', 'decimal',
        'fractions', 'statistics', 'heapq', 'bisect', 'array', 'weakref', 'pprint',
        'struct', 'codecs', 'argparse', 'logging', 'subprocess', 'socket', 'asyncio',
        'threading', 'queue', 'pickle', 'sqlite3', 'csv', 'configparser', 'hashlib',
        'hmac', 'secrets', 'html', 'xml', 'base64', 'difflib', 'urllib', 'http',
        'uuid', 'tempfile', 'shutil', 'glob', 'fnmatch', 'zipfile', 'tarfile', 'gzip',
        'mock', 'unittest', 'pytest_mock', 'freezegun', 'responses', 'requests_mock',
    }
    samples = []
    for sample in itertools.islice(ds, 500_000):
        content = sample.get('content', '')
        # Check for pytest patterns but reject relative imports (missing helper files)
        if 'import pytest' in content and 'def test_' in content and 'assert ' in content:
            if 'from .' in content or 'import .' in content:
                continue  # Skip files with relative imports
            # All imports must be in safe_packages
            imports = re.findall(r'^(?:from|import)\s+(\w+)', content, re.MULTILINE)
            if all(imp in safe_packages for imp in imports):
                samples.append({'text': content, 'path': sample.get('path', ''), 'repo': sample.get('repository_name', '')})
                if len(samples) >= limit:
                    break
    return samples

def create_pytest_test_sh():
    return '''#!/bin/bash
set -e
mkdir -p /logs/verifier
cleanup() { if [ $? -eq 0 ]; then echo "1" > /logs/verifier/reward.txt; else echo "0" > /logs/verifier/reward.txt; fi; }
trap cleanup EXIT

# Create virtual environment if needed (fixes PEP 668 on Python 3.12+)
if [ ! -d /app/.venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv /app/.venv
fi
source /app/.venv/bin/activate

if ! command -v pytest &> /dev/null; then
    pip install --quiet pytest
fi

# Auto-detect and install missing imports from test file
echo "Checking for missing dependencies..."
MISSING=$(python3 -c "
import re, sys
with open('/tests/test_solution.py') as f:
    for line in f:
        m = re.match(r'^(?:from|import)\\s+(\\w+)', line.strip())
        if m:
            mod = m.group(1)
            if mod in ('pytest', 'sys', 'os', 're', 'json', 'math', 'collections', 'itertools', 'functools', 'typing', 'unittest', 'pathlib', 'datetime', 'time', 'copy', 'io', 'string', 'random', 'abc', 'contextlib', 'warnings', 'operator', 'dataclasses', 'enum', 'textwrap', 'inspect', 'types', 'numbers', 'decimal', 'fractions', 'statistics', 'heapq', 'bisect', 'array', 'weakref', 'pprint', 'reprlib', 'struct', 'codecs', 'locale', 'gettext', 'argparse', 'optparse', 'logging', 'errno', 'signal', 'subprocess', 'socket', 'select', 'selectors', 'asyncio', 'threading', 'multiprocessing', 'concurrent', 'queue', 'sched', 'contextvars', 'pickle', 'shelve', 'dbm', 'sqlite3', 'csv', 'configparser', 'tomllib', 'netrc', 'plistlib', 'hashlib', 'hmac', 'secrets', 'html', 'xml', 'base64', 'binascii', 'quopri', 'uu', 'difflib', 'cgi', 'cgitb', 'wsgiref', 'urllib', 'http', 'ftplib', 'poplib', 'imaplib', 'smtplib', 'uuid', 'telnetlib', 'mailbox', 'mimetypes', 'email', 'mailcap', 'audioop', 'wave', 'colorsys', 'imghdr', 'sndhdr', 'getpass', 'curses', 'platform', 'ctypes', 'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'shutil', 'glob', 'fnmatch', 'linecache', 'fileinput', 'tempfile', 'filecmp', 'stat', 'test'):
                continue
            try:
                __import__(mod)
            except ImportError:
                print(mod)
" 2>/dev/null)
[ -n "$MISSING" ] && pip install --quiet $MISSING 2>/dev/null || true

cd /app
pytest /tests/test_solution.py -v --tb=short 2>&1 | tee /logs/verifier/pytest_output.txt
PYTEST_EXIT=${PIPESTATUS[0]}
if [ $PYTEST_EXIT -eq 0 ]; then exit 0; else exit 1; fi
'''

# =============================================================================
# Ruby Generator with build-essential fix
# =============================================================================

def filter_ruby_tests(limit: int) -> List[Dict]:
    ds = load_dataset('bigcode/the-stack', data_dir='data/ruby', split='train', streaming=True)
    # Helper files and frameworks to reject
    helper_requires = ['spec_helper', 'test_helper', 'rails_helper', 'support/', '../']
    samples = []
    for sample in itertools.islice(ds, 500_000):
        content = sample.get('content', '')
        has_rspec = 'RSpec.describe' in content or ('describe' in content and 'it ' in content)
        has_minitest = 'Minitest' in content or 'def test_' in content
        if (has_rspec or has_minitest) and ('expect(' in content or 'assert' in content):
            # Reject tests with external helper requires
            has_helper = any(f"require '{h}" in content or f'require "{h}' in content for h in helper_requires)
            if has_helper:
                continue
            # Reject Rails tests
            if 'ActiveRecord' in content or 'ActionController' in content or 'rails' in content.lower():
                continue
            # Reject relative requires
            if 'require_relative' in content:
                continue
            samples.append({'text': content, 'path': sample.get('path', ''), 'repo': sample.get('repository_name', '')})
            if len(samples) >= limit:
                break
    return samples

def create_ruby_dockerfile():
    return """FROM ruby:3.2-slim

WORKDIR /app

# Install build tools required for native gem extensions, bash for Harbor agent, and bsdutils for script command
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    ruby-dev \\
    libffi-dev \\
    bash \\
    bsdutils \\
    && rm -rf /var/lib/apt/lists/*

RUN gem install rspec minitest --no-document
"""

def create_ruby_test_sh():
    return '''#!/bin/bash
set -e
mkdir -p /logs/verifier
cleanup() { if [ $? -eq 0 ]; then echo "1" > /logs/verifier/reward.txt; else echo "0" > /logs/verifier/reward.txt; fi; }
trap cleanup EXIT
cd /app
if [ -f Gemfile ]; then bundle install --quiet 2>/dev/null || true; fi

# Create a helper that loads all .rb files from /app before running tests
cat > /tmp/load_app.rb << 'RUBY'
$LOAD_PATH.unshift('/app')
Dir.glob('/app/**/*.rb').sort.each do |file|
  begin
    require file
  rescue LoadError, NameError => e
  end
end
RUBY

if grep -q "RSpec" /tests/test_solution.rb 2>/dev/null; then
    rspec --require /tmp/load_app.rb /tests/test_solution.rb 2>&1 | tee /logs/verifier/test_output.txt
else
    ruby -r /tmp/load_app.rb /tests/test_solution.rb 2>&1 | tee /logs/verifier/test_output.txt
fi
exit ${PIPESTATUS[0]}
'''

# =============================================================================
# Jest Generator with jest.config.js fix
# =============================================================================

def filter_jest(limit: int) -> List[Dict]:
    ds = load_dataset('bigcode/the-stack', data_dir='data/javascript', split='train', streaming=True)
    samples = []
    for sample in itertools.islice(ds, 500_000):
        content = sample.get('content', '')
        if ('describe(' in content or 'it(' in content or 'test(' in content) and 'expect(' in content:
            if len(re.findall(r'\b(it|test)\s*\(', content)) >= 2:
                # Reject ES6 imports (needs Babel)
                if re.search(r'^import\s+', content, re.MULTILINE):
                    continue
                # Reject JSX syntax (needs React preset)
                if '</' in content and re.search(r'<[A-Z][a-zA-Z]*', content):
                    continue
                # Reject AngularJS patterns
                if 'beforeEach(module(' in content or 'angular.mock' in content:
                    continue
                # Reject relative imports to project modules
                if re.search(r"require\(['\"]\.\.?/(?!tests)", content):
                    continue
                # Reject DOM/jsdom tests
                if 'document.' in content or 'window.' in content:
                    continue
                samples.append({'text': content, 'path': sample.get('path', ''), 'repo': sample.get('repository_name', '')})
                if len(samples) >= limit:
                    break
    return samples

def create_jest_dockerfile():
    return """FROM node:20-slim

WORKDIR /app

# Install bsdutils for script command needed by terminus-2 agent
RUN apt-get update && apt-get install -y --no-install-recommends bsdutils && rm -rf /var/lib/apt/lists/*

# Install test frameworks globally
RUN npm install -g jest mocha chai

# Create minimal jest.config.js so Jest can find tests
RUN echo 'module.exports = { testEnvironment: "node", testMatch: ["**/tests/**/*.js", "**/*.test.js", "**/*.spec.js"] };' > /app/jest.config.js
"""

def create_jest_test_sh():
    return '''#!/bin/bash
set -e
mkdir -p /logs/verifier
cleanup() { if [ $? -eq 0 ]; then echo "1" > /logs/verifier/reward.txt; else echo "0" > /logs/verifier/reward.txt; fi; }
trap cleanup EXIT
cd /app

# Create jest.config.js at runtime
cat > /app/jest.config.js << 'EOF'
module.exports = {
  testEnvironment: "node",
  rootDir: "/app",
  testMatch: ["**/tests/**/*.js", "**/*.test.js"],
  moduleDirectories: ["node_modules", "/app", "/app/lib"]
};
EOF

if [ -f package.json ]; then npm install --silent 2>/dev/null || true; fi

# Auto-detect and install missing npm modules from test file
echo "Checking for missing dependencies..."
REQUIRES=$(grep -oP "require\\(['\"]\\K[^'\"@./][^'\"]*" /tests/test_solution.js 2>/dev/null | sort -u || true)
for pkg in $REQUIRES; do
    if [ ! -d "/app/node_modules/$pkg" ] && [ ! -d "/usr/local/lib/node_modules/$pkg" ]; then
        echo "Installing $pkg..."
        npm install --silent "$pkg" 2>/dev/null || true
    fi
done

# Create symlinks so relative requires from /tests/ can find /app/lib
if [ -d /app/lib ]; then ln -sf /app/lib /lib 2>/dev/null || true; fi

# Copy test file to /app/tests for Jest to find it
mkdir -p /app/tests
cp /tests/test_solution.js /app/tests/

echo "Running tests..."
timeout 300 npx jest /app/tests/test_solution.js --config /app/jest.config.js --no-coverage 2>&1 | tee /logs/verifier/test_output.txt
exit ${PIPESTATUS[0]}
'''

# =============================================================================
# PHP Generator with phpunit.xml fix
# =============================================================================

def filter_php_tests(limit: int) -> List[Dict]:
    ds = load_dataset('bigcode/the-stack', data_dir='data/php', split='train', streaming=True)
    samples = []
    for sample in itertools.islice(ds, 500_000):
        content = sample.get('content', '')
        has_phpunit = 'PHPUnit' in content or 'extends TestCase' in content
        has_test_method = 'function test' in content.lower() or '@test' in content
        has_assertions = any(a in content for a in ['assertEquals', 'assertTrue', 'assertFalse', 'assertSame', 'assertNull'])
        if has_phpunit and has_test_method and has_assertions:
            if len(re.findall(r'function\s+test\w+', content)) >= 2 or content.count('@test') >= 2:
                samples.append({'text': content, 'path': sample.get('path', ''), 'repo': sample.get('repository_name', '')})
                if len(samples) >= limit:
                    break
    return samples

def create_php_dockerfile():
    return r"""FROM php:8.2-cli

WORKDIR /app

# Install required tools including bash for Harbor agent
RUN apt-get update && apt-get install -y unzip git bash bsdutils && rm -rf /var/lib/apt/lists/*

# Install Composer
RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer

# Install PHPUnit globally
RUN composer global require phpunit/phpunit
ENV PATH="${PATH}:/root/.composer/vendor/bin"

# Create autoload file for test classes using heredoc to avoid escaping issues
RUN cat > /app/autoload.php << 'AUTOLOAD'
<?php
spl_autoload_register(function($class) {
    $file = "/app/" . str_replace("\\", "/", $class) . ".php";
    if (file_exists($file)) {
        require $file;
    }
});
AUTOLOAD
"""

def create_php_test_sh():
    return '''#!/bin/bash
set -e
mkdir -p /logs/verifier
cleanup() { if [ $? -eq 0 ]; then echo "1" > /logs/verifier/reward.txt; else echo "0" > /logs/verifier/reward.txt; fi; }
trap cleanup EXIT
cd /app

# Install dependencies if composer.json exists
if [ -f composer.json ]; then
    composer install --quiet 2>/dev/null || true
fi

# Create phpunit.xml for better test discovery
cat > /app/phpunit.xml << 'EOF'
<?xml version="1.0"?>
<phpunit bootstrap="/app/autoload.php">
  <testsuites>
    <testsuite name="Tests">
      <directory>/tests</directory>
    </testsuite>
  </testsuites>
</phpunit>
EOF

# Run PHPUnit with configuration
echo "Running PHPUnit tests..."
timeout 300 phpunit --configuration /app/phpunit.xml 2>&1 | tee /logs/verifier/test_output.txt
exit ${PIPESTATUS[0]}
'''

# =============================================================================
# Task Generation
# =============================================================================

PROMPTS = {
    "go": "Given Go test code, create a task description for what needs to be implemented. The solution goes in /app/.\n\nTest code:\n{{text}}\n\nTask (no preamble):",
    "pytest": "Given pytest code, create a task description for what needs to be implemented. The solution goes in /app/.\n\nTest code:\n{{text}}\n\nTask (no preamble):",
    "ruby": "Given Ruby test code, create a task description for what needs to be implemented. The solution goes in /app/.\n\nTest code:\n{{text}}\n\nTask (no preamble):",
    "jest": "Given Jest test code, create a task description for what needs to be implemented. The solution goes in /app/.\n\nTest code:\n{{text}}\n\nTask (no preamble):",
    "php": "Given PHP PHPUnit test code, create a task description for what needs to be implemented. The solution goes in /app/.\n\nTest code:\n{{text}}\n\nTask (no preamble):",
}


def create_task_dir(output_dir: Path, task_id: int, instruction: str, test_code: str,
                    prefix: str, dockerfile_fn, test_sh_fn, test_filename: str):
    task_dir = output_dir / f"{prefix}-{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(dockerfile_fn(), encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(test_sh_fn(), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)
    (tests_dir / test_filename).write_text(test_code, encoding="utf-8")

    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    return task_dir


def generate_dataset(name: str, filter_fn, dockerfile_fn, test_sh_fn, test_filename: str, prompt: str):
    print(f"\n{'='*60}")
    print(f"Generating {name} dataset")
    print(f"{'='*60}")

    # Filter
    print("Filtering samples...")
    samples = filter_fn(LIMIT)
    print(f"Found {len(samples)} samples")

    if not samples:
        return None

    # Synthesize
    print("Synthesizing task descriptions...")
    dataset = Dataset.from_list(samples)
    result = run_completions(
        dataset, model=MODEL, map_type="chat",
        map_config={"user_message": prompt, "output_column": "task_description"},
        max_requests_per_minute=500, max_tokens_per_minute=1_000_000,
        require_all_responses=False,
    )
    descriptions = result.dataset["task_description"]
    print(f"Generated {len(descriptions)} descriptions")

    # Create tasks
    print("Creating task directories...")
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{name}_test_"))
    for i, (sample, desc) in enumerate(tqdm(zip(samples, descriptions), total=len(samples))):
        create_task_dir(temp_dir, i, desc, sample["text"], f"stack-{name}",
                       dockerfile_fn, test_sh_fn, test_filename)

    # Upload
    print("Uploading to HuggingFace...")
    repo_id = f"DCAgent/exp_rpt_stack-{name}{TEST_SUFFIX}"
    repo_url = upload_tasks_to_hf(str(temp_dir), repo_id)
    print(f"Uploaded to: {repo_url}")

    return repo_id


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=["go", "pytest", "ruby", "jest", "php"],
                       help="Languages to generate")
    args = parser.parse_args()

    generators = {
        "go": (filter_go_tests, create_go_dockerfile, create_go_test_sh, "solution_test.go", PROMPTS["go"]),
        "pytest": (filter_pytest, create_standard_dockerfile, create_pytest_test_sh, "test_solution.py", PROMPTS["pytest"]),
        "ruby": (filter_ruby_tests, create_ruby_dockerfile, create_ruby_test_sh, "test_solution.rb", PROMPTS["ruby"]),
        "jest": (filter_jest, create_jest_dockerfile, create_jest_test_sh, "test_solution.js", PROMPTS["jest"]),
        "php": (filter_php_tests, create_php_dockerfile, create_php_test_sh, "TestSolution.php", PROMPTS["php"]),
    }

    results = []
    for lang in args.languages:
        if lang not in generators:
            print(f"Unknown: {lang}")
            continue
        filter_fn, dockerfile_fn, test_sh_fn, test_file, prompt = generators[lang]
        try:
            repo_id = generate_dataset(lang, filter_fn, dockerfile_fn, test_sh_fn, test_file, prompt)
            results.append((lang, repo_id, "success"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((lang, None, str(e)))

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for lang, repo_id, status in results:
        print(f"{lang}: {repo_id if status == 'success' else status}")

    print("\nDatasets for pass rate testing:")
    for lang, repo_id, status in results:
        if status == "success":
            print(f"  {repo_id}")


if __name__ == "__main__":
    main()
