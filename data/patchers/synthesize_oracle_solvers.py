#!/usr/bin/env python3
"""
Synthesize oracle solvers (solution/solve.sh) for Nemotron-Terminal-Synthetic-Tasks.

For each task directory, reads ONLY instruction.md (never test files) and uses
an LLM to generate a bash script that correctly completes the task.  The verifier
files (test.sh, test_outputs.py) are intentionally withheld to prevent the LLM
from cheating by hardcoding expected outputs or modifying the test suite.

After running this script, validate the generated solvers with:
    python scripts/daytona/validate_and_upload_from_hf.py \
        --extract_dir /path/to/patched_tasks \
        --oracle_check_only \
        --target_repo open-thoughts/nemotron-synthetic-tasks-rl

Usage:
    # Generate oracle solvers for a directory of patched tasks:
    python synthesize_oracle_solvers.py /path/to/patched_tasks

    # Limit to first N tasks (useful for testing):
    python synthesize_oracle_solvers.py /path/to/patched_tasks --limit 20

    # Use a specific model and concurrency:
    python synthesize_oracle_solvers.py /path/to/patched_tasks --model gpt-5-mini --workers 32

    # Dry run (show what would be generated without writing):
    python synthesize_oracle_solvers.py /path/to/patched_tasks --dry-run
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Linux terminal engineer. Your job is to write a bash script \
that correctly completes a given task inside a Docker container running Ubuntu.

Rules:
- Output ONLY a bash script (starting with #!/bin/bash), no markdown fences, \
no explanation.
- The script must actually solve the task — do NOT hardcode expected outputs \
or cheat.
- Use standard Unix tools, Python 3, or whatever is appropriate for the task.
- Assume the working directory is /app. Write output files to /app/ unless \
the task specifies otherwise.
- If data files are needed, they are available in /setup_files/ — copy them \
to /app/ first with: cp -r /setup_files/. /app/
- Keep the script concise and correct.
"""

USER_PROMPT_TEMPLATE = """\
Complete the following task by writing a bash script (solve.sh).

--- TASK ---
{instruction}
--- END TASK ---

Output ONLY the bash script, starting with #!/bin/bash.
"""


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def _call_llm(instruction: str, model: str, client) -> str:
    """Call the OpenAI API and return the generated script text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(instruction=instruction)},
        ],
        max_completion_tokens=128000,
    )
    return response.choices[0].message.content.strip()


def _clean_script(raw: str) -> str:
    """Strip markdown code fences if the LLM added them despite instructions."""
    if raw.startswith("```"):
        lines = raw.splitlines()
        # Drop first line (```bash or ```) and last line (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        raw = "\n".join(inner)
    if not raw.startswith("#!"):
        raw = "#!/bin/bash\n" + raw
    return raw


def synthesize_one(
    task_dir: Path,
    *,
    model: str,
    client,
    overwrite: bool = False,
    dry_run: bool = False,
    max_retries: int = 3,
) -> dict[str, object]:
    """Generate solution/solve.sh for a single task directory.

    Returns a result dict with keys: task, status, error.
    Possible statuses: "skipped", "dry_run", "ok", "error"
    """
    result: dict[str, object] = {"task": task_dir.name}

    instruction_path = task_dir / "instruction.md"
    if not instruction_path.exists():
        result["status"] = "error"
        result["error"] = "no instruction.md"
        return result

    solution_dir = task_dir / "solution"
    solve_path = solution_dir / "solve.sh"

    if solve_path.exists() and not overwrite:
        result["status"] = "skipped"
        return result

    if dry_run:
        result["status"] = "dry_run"
        return result

    instruction = instruction_path.read_text(encoding="utf-8")

    # Retry loop with exponential backoff
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            raw = _call_llm(instruction, model, client)
            script = _clean_script(raw)
            solution_dir.mkdir(exist_ok=True)
            solve_path.write_text(script, encoding="utf-8")
            solve_path.chmod(0o755)
            result["status"] = "ok"
            return result
        except Exception as exc:
            last_error = exc
            wait = 2 ** attempt
            time.sleep(wait)

    result["status"] = "error"
    result["error"] = str(last_error)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize oracle solvers for Nemotron-Terminal-Synthetic-Tasks"
    )
    parser.add_argument(
        "tasks_dir",
        type=Path,
        help="Directory containing patched task folders (each with instruction.md)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model to use (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel API calls (default: 32)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N tasks (for testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing solution/solve.sh files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without calling the API",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )
    args = parser.parse_args()

    # ---- Validate input directory ----
    if not args.tasks_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.tasks_dir}")

    task_dirs = sorted(
        d for d in args.tasks_dir.iterdir()
        if d.is_dir() and (d / "instruction.md").exists()
    )
    if not task_dirs:
        raise SystemExit(f"No tasks found in {args.tasks_dir}")

    if args.limit:
        task_dirs = task_dirs[: args.limit]

    print(f"Found {len(task_dirs)} tasks in {args.tasks_dir}")
    if args.dry_run:
        print("Dry run — no API calls will be made")

    # ---- Set up OpenAI client ----
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("openai is required. Run: pip install openai")

        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "OpenAI API key not found. Set OPENAI_API_KEY env var or pass --api-key."
            )
        client = OpenAI(api_key=api_key)
    else:
        client = None

    # ---- Run generation in parallel ----
    counts = {"ok": 0, "skipped": 0, "dry_run": 0, "error": 0}
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                synthesize_one,
                td,
                model=args.model,
                client=client,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            ): td
            for td in task_dirs
        }

        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            status = result["status"]
            counts[status] = counts.get(status, 0) + 1
            if status == "error":
                errors.append(f"  {result['task']}: {result.get('error', '?')}")
            if completed % 100 == 0 or completed == total:
                print(
                    f"  [{completed}/{total}] ok={counts['ok']} "
                    f"skipped={counts['skipped']} error={counts['error']}"
                )

    # ---- Report ----
    print("\nDone:")
    print(f"  Generated : {counts['ok']}")
    print(f"  Skipped   : {counts['skipped']} (already had solve.sh)")
    if args.dry_run:
        print(f"  Dry run   : {counts['dry_run']}")
    print(f"  Errors    : {counts['error']}")
    if errors:
        print("\nFailed tasks:")
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    print(
        "\nNext step: validate with\n"
        "  python scripts/daytona/validate_and_upload_from_hf.py \\\n"
        f"    --extract_dir {args.tasks_dir} \\\n"
        "    --oracle_check_only \\\n"
        "    --target_repo open-thoughts/nemotron-synthetic-tasks-rl"
    )


if __name__ == "__main__":
    main()
