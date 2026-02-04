#!/usr/bin/env python3
"""
Generate tasks from E2EGit dataset (43,670 end-to-end web tests).

E2EGit contains automated Web GUI tests from real repositories using:
- Selenium
- Playwright
- Cypress
- Puppeteer

These are integration tests that span multiple components and require
understanding of web frameworks and browser testing patterns.
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
    download_e2egit,
    download_from_zenodo,
    get_cache_dir,
    load_e2egit_from_sqlite,
)
from data.completions import run_completions
from data.commons import (
    TEST_TO_INSTRUCTION_PROMPT,
    create_harbor_task_directory_generic,
    create_generic_test_sh,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"

# Dockerfile with browser testing capabilities
E2EGIT_DOCKERFILE = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for browser testing
RUN apt-get update && apt-get install -y \\
    wget \\
    gnupg \\
    curl \\
    unzip \\
    xvfb \\
    libxi6 \\
    libgconf-2-4 \\
    default-jdk \\
    chromium \\
    chromium-driver \\
    firefox-esr \\
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for Cypress/Playwright
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \\
    && apt-get install -y nodejs \\
    && rm -rf /var/lib/apt/lists/*

# Install Python testing packages
RUN pip install --no-cache-dir \\
    pytest \\
    pytest-timeout \\
    selenium \\
    playwright \\
    webdriver-manager \\
    beautifulsoup4 \\
    requests

# Install Playwright browsers
RUN playwright install chromium firefox

# Install Node.js testing packages
RUN npm install -g cypress puppeteer jest

# Set up display for headless testing
ENV DISPLAY=:99

RUN mkdir -p /logs/verifier
"""

# Alternative lightweight Dockerfile
E2EGIT_DOCKERFILE_LIGHT = """FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \\
    pytest \\
    pytest-timeout \\
    selenium \\
    beautifulsoup4 \\
    requests

RUN mkdir -p /logs/verifier
"""


def get_e2egit_dockerfile(framework: str = "selenium", lightweight: bool = True) -> str:
    """Get appropriate Dockerfile for the testing framework."""
    if lightweight:
        return E2EGIT_DOCKERFILE_LIGHT
    return E2EGIT_DOCKERFILE


def create_e2egit_test_sh(framework: str = "selenium") -> str:
    """Create test.sh for E2EGit tasks."""
    # Framework-specific test commands
    test_commands = {
        "selenium": "pytest /tests/test_e2e.py -v --tb=short --timeout=120",
        "playwright": "pytest /tests/test_e2e.py -v --tb=short --timeout=120",
        "cypress": "npx cypress run --spec /tests/test_e2e.cy.js",
        "puppeteer": "npx jest /tests/test_e2e.js --verbose --testTimeout=120000",
    }

    test_cmd = test_commands.get(framework.lower(), test_commands["selenium"])

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

cd /app

# Start virtual display for headless testing
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

echo "Running E2E tests ({framework})..."
{test_cmd} 2>&1 | tee /logs/verifier/test_output.txt

exit ${{PIPESTATUS[0]}}
'''


def detect_framework(test_code: str) -> str:
    """Detect which E2E testing framework is being used."""
    test_lower = test_code.lower()

    if "from selenium" in test_lower or "webdriver" in test_lower:
        return "selenium"
    elif "from playwright" in test_lower or "playwright" in test_lower:
        return "playwright"
    elif "cy." in test_lower or "cypress" in test_lower:
        return "cypress"
    elif "puppeteer" in test_lower or "page.goto" in test_lower:
        return "puppeteer"

    return "selenium"  # Default


def load_e2egit(limit: int = LIMIT, local_db_path: Path = None) -> List[Dict]:
    """
    Load E2EGit dataset.

    E2EGit was presented at MSR 2025 and contains E2E tests
    from repositories using Selenium, Playwright, Cypress, and Puppeteer.

    The dataset is available as a SQLite database from Zenodo:
    https://zenodo.org/records/14234731

    Args:
        limit: Maximum samples to load
        local_db_path: Path to locally downloaded E2EGit.db SQLite file
    """
    print(f"Loading E2EGit dataset...")

    # Option 1: Load from local SQLite file if provided
    if local_db_path and Path(local_db_path).exists():
        print(f"Loading from local SQLite database: {local_db_path}")
        try:
            samples = load_e2egit_from_sqlite(Path(local_db_path), limit=limit)
            # Enrich samples with framework detection
            for sample in samples:
                if not sample.get("framework"):
                    sample["framework"] = detect_framework(sample.get("test_code", ""))
            return samples
        except Exception as e:
            print(f"Error loading from local database: {e}")

    # Option 2: Check cache for previously downloaded database
    cache_dir = get_cache_dir()
    cached_db = cache_dir / "zenodo_14234731" / "E2EGit.db"
    if cached_db.exists():
        print(f"Loading from cached database: {cached_db}")
        try:
            samples = load_e2egit_from_sqlite(cached_db, limit=limit)
            for sample in samples:
                if not sample.get("framework"):
                    sample["framework"] = detect_framework(sample.get("test_code", ""))
            return samples
        except Exception as e:
            print(f"Error loading from cache: {e}")

    # Option 3: Try HuggingFace sources
    repo_names = [
        "e2egit/e2egit",
        "msr2025/e2egit",
        "e2e-tests/e2egit",
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
        print("Source: https://zenodo.org/records/14234731")
        try:
            db_path = download_e2egit()
            samples = load_e2egit_from_sqlite(db_path, limit=limit)
            for sample in samples:
                if not sample.get("framework"):
                    sample["framework"] = detect_framework(sample.get("test_code", ""))
            return samples
        except Exception as e:
            print(f"Zenodo download failed: {e}")

        # Fallback: Create sample data for demonstration
        print("")
        print("To load the full E2EGit dataset:")
        print("  1. Download E2EGit.db from https://zenodo.org/records/14234731")
        print("  2. Run with --local-db /path/to/E2EGit.db")
        print("")
        print("Generating sample E2E test data for demonstration...")
        return _create_sample_e2e_data(limit)

    samples = []
    print(f"Collecting up to {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        test_code = sample.get("test_code", sample.get("code", ""))
        repo = sample.get("repo", sample.get("repository", ""))

        if not test_code:
            continue

        framework = detect_framework(test_code)

        samples.append({
            "test_code": test_code,
            "repo": repo,
            "file_path": sample.get("file_path", sample.get("path", "")),
            "framework": framework,
            "language": sample.get("language", "python" if framework in ["selenium", "playwright"] else "javascript"),
            "url_patterns": sample.get("url_patterns", []),
            "selectors": sample.get("selectors", []),
        })

    print(f"Loaded {len(samples)} E2E test samples")
    return samples


def _create_sample_e2e_data(limit: int) -> List[Dict]:
    """Create sample E2E test data for demonstration."""
    samples = [
        {
            "test_code": '''from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_login_page():
    driver = webdriver.Chrome()
    driver.get("http://localhost:3000/login")

    username_input = driver.find_element(By.ID, "username")
    password_input = driver.find_element(By.ID, "password")
    submit_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")

    username_input.send_keys("testuser")
    password_input.send_keys("password123")
    submit_button.click()

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "dashboard"))
    )

    assert "Dashboard" in driver.title
    driver.quit()''',
            "repo": "example/web-app",
            "framework": "selenium",
            "language": "python",
        },
        {
            "test_code": '''import { test, expect } from '@playwright/test';

test('shopping cart flow', async ({ page }) => {
    await page.goto('http://localhost:3000/products');

    // Add item to cart
    await page.click('.product-card:first-child .add-to-cart');

    // Check cart badge
    const cartBadge = page.locator('.cart-badge');
    await expect(cartBadge).toHaveText('1');

    // Go to cart
    await page.click('.cart-icon');
    await expect(page).toHaveURL(/.*cart/);

    // Verify item in cart
    const cartItems = page.locator('.cart-item');
    await expect(cartItems).toHaveCount(1);
});''',
            "repo": "example/ecommerce",
            "framework": "playwright",
            "language": "javascript",
        },
        {
            "test_code": '''describe('Form Validation', () => {
    beforeEach(() => {
        cy.visit('/register')
    })

    it('should show error for invalid email', () => {
        cy.get('#email').type('invalid-email')
        cy.get('#submit').click()
        cy.get('.error-message').should('contain', 'Invalid email')
    })

    it('should show error for short password', () => {
        cy.get('#password').type('123')
        cy.get('#submit').click()
        cy.get('.error-message').should('contain', 'at least 8 characters')
    })
})''',
            "repo": "example/forms",
            "framework": "cypress",
            "language": "javascript",
        },
    ]

    return samples[:limit]


def create_instruction_from_e2e_test(sample: Dict, generated_instruction: str = None) -> str:
    """Create task instruction from E2E test code."""
    test_code = sample.get("test_code", "")
    framework = sample.get("framework", "selenium")
    language = sample.get("language", "python")

    if generated_instruction:
        main_instruction = generated_instruction
    else:
        main_instruction = "Implement the web application functionality required to pass the following E2E tests."

    instruction = f"""# End-to-End Test Implementation Task

## Testing Framework
{framework.capitalize()}

## Language
{language.capitalize()}

## Task Description

{main_instruction}

## E2E Test Code

The following test code defines the expected behavior of the web application:

```{language}
{test_code}
```

## Your Task

1. Analyze the E2E test to understand the required functionality
2. Implement the necessary components/pages to make the tests pass
3. Ensure proper element IDs, classes, and selectors match the test expectations
4. Handle user interactions (clicks, form submissions, navigation) correctly

Place your implementation in `/app/`
"""

    return instruction


def get_test_filename(framework: str, language: str) -> str:
    """Get appropriate test filename for framework."""
    if framework == "cypress":
        return "test_e2e.cy.js"
    elif framework == "puppeteer":
        return "test_e2e.js"
    elif language == "python":
        return "test_e2e.py"
    return "test_e2e.js"


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    sample: Dict,
    instruction: str,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for an E2EGit sample."""
    framework = sample.get("framework", "selenium")
    language = sample.get("language", "python")
    test_filename = get_test_filename(framework, language)

    test_files = {
        test_filename: sample.get("test_code", ""),
    }

    metadata = {
        "source": "e2egit",
        "repo": sample.get("repo", ""),
        "file_path": sample.get("file_path", ""),
        "framework": framework,
        "language": language,
    }

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_e2egit_dockerfile(framework, lightweight=True),
        test_sh=create_e2egit_test_sh(framework),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "e2egit"
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction) in enumerate(tqdm(
        zip(samples, instructions),
        total=len(samples),
        desc="Creating tasks"
    )):
        create_harbor_task(temp_dir, i, sample, instruction, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(limit: int = LIMIT, upload: bool = True, local_db_path: Path = None) -> None:
    """Main pipeline for generating E2EGit tasks."""

    print(f"Step 1: Loading E2EGit dataset...")
    samples = load_e2egit(limit=limit, local_db_path=local_db_path)
    print(f"  -> {len(samples)} E2E test samples loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating task descriptions from E2E tests...")
    dataset = Dataset.from_list([{"text": s["test_code"]} for s in samples])

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

    print("\nStep 3: Generating harbor task directories...")
    dataset_prefix = "e2egit"
    task_dir = generate_tasks(samples, instructions, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 4: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} E2EGit tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from E2EGit dataset")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")
    parser.add_argument("--local-db", type=Path,
                        help="Path to locally downloaded E2EGit.db SQLite file (from zenodo.org/records/14234731)")

    args = parser.parse_args()
    main(limit=args.limit, upload=not args.no_upload, local_db_path=args.local_db)
