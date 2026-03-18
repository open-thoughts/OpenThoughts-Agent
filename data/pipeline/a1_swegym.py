#!/usr/bin/env python3
"""
Generate Harbor-compatible tasks from the SWE-Gym dataset.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys
from textwrap import dedent
from typing import Dict, List, Optional, Sequence, Tuple

from datasets import Dataset, load_dataset

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.commons import (  # type: ignore  # pylint: disable=wrong-import-position
    create_task_directory_unified,
    finalize_dataset_output,
    upload_tasks_to_hf,
)

if __package__ in (None, ""):
    from data.swegym import DATASET_NAME  # type: ignore
else:
    from . import DATASET_NAME


# --------------------------------------------------------------------------- #
# Argument handling


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert rows from the SWE-Gym dataset into Harbor sandboxes "
            "with instructions, environments, oracle patches, and verifiers."
        )
    )
    return add_swegym_args(parser)


def add_swegym_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load.")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of tasks to materialize (<=0 means no limit).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting index inside the split.",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="swegym",
        help="Directory name prefix for generated tasks.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional final directory for the dataset.",
    )
    parser.add_argument(
        "--target-repo",
        type=str,
        default=None,
        help="Optional Hugging Face repo to upload into.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token used for uploads (defaults to env token).",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Upload the dataset to a private Hugging Face repo.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading even if --target-repo is provided.",
    )
    return parser


# --------------------------------------------------------------------------- #
# Rendering helpers


BASE_DOCKERFILE_TEMPLATE = """\
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /testbed

RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
        git \\
        python3 \\
        python3-pip \\
        python3-venv \\
        build-essential \\
        pkg-config \\
        ca-certificates \\
        curl \\
        jq \\
    && rm -rf /var/lib/apt/lists/*

RUN git clone {repo_url} repo && cd repo && git checkout {base_commit}

ENV PATH=\"/root/.local/bin:$PATH\"
"""

TASK_TOML_CONTENT = """\
version = "1.0"

[agent]
timeout_sec = 1800.0

[metadata]
author_name = "OT-Agent + OpenHands"
author_email = "bf996@nyu.edu"
difficulty = "medium"
category = "bugfix"
tags = [
    "swe",
    "bugfix",
]

[verifier]
restart_environment = false
timeout_sec = 120.0
"""


def _normalize_tests(raw: Optional[Sequence[str]]) -> List[str]:
    normalized: List[str] = []
    if not raw:
        return normalized
    for entry in raw:
        if not entry:
            continue
        value = entry.strip()
        if value and value.lower() != "nan":
            normalized.append(value)
    return normalized


def _render_instruction(
    row: Dict[str, str],
    pass_tests: Sequence[str],
    fail_tests: Sequence[str],
) -> str:
    repo_url = f"https://github.com/{row['repo']}"
    instruction_lines = [
        "# Bug Fix Task",
        "",
        f"- Repository: `{row['repo']}`",
        f"- Source commit: `{row.get('base_commit', 'unknown')}`",
        f"- Dataset instance: `{row.get('instance_id', 'n/a')}`",
        f"- Upstream issue version: `{row.get('version', 'n/a')}`",
        f"- Reference repo URL: {repo_url}",
        "",
        "## Problem Statement",
        row.get("problem_statement", "").strip(),
    ]

    if pass_tests:
        instruction_lines.extend(
            [
                "",
                "## Tests that must keep passing",
                *[f"- `{test}`" for test in pass_tests],
            ]
        )

    if fail_tests:
        instruction_lines.extend(
            [
                "",
                "## Tests currently failing (should pass after your fix)",
                *[f"- `{test}`" for test in fail_tests],
            ]
        )

    instruction_lines.extend(
        [
            "",
            "Apply code changes directly inside `/testbed/repo` and use `pytest` "
            "to run any of the above test targets. Focus on making the failing "
            "tests succeed without regressing the existing passing suite.",
        ]
    )

    return "\n".join(instruction_lines).strip() + "\n"


def _render_dockerfile(row: Dict[str, str]) -> str:
    repo = row.get("repo")
    if not repo:
        raise ValueError("Missing 'repo' in row; cannot render Dockerfile.")
    repo_url = f"https://github.com/{repo}.git"
    base_commit = row.get("base_commit", "main")
    return BASE_DOCKERFILE_TEMPLATE.format(repo_url=repo_url, base_commit=base_commit)


def _render_solution_script(test_patch: str) -> Optional[str]:
    if not test_patch or not test_patch.strip():
        return None

    return dedent(
        f"""\
        #!/usr/bin/env bash
        set -Eeuo pipefail

        cd /testbed/repo

        patch_file="$(mktemp /tmp/swegym-test-patch-XXXX.diff)"
        cat <<'PATCH_EOF' > "$patch_file"
        {test_patch.rstrip()}
        PATCH_EOF

        git apply --whitespace=nowarn --apply "$patch_file"
        """
    )


def _format_bash_array(items: Sequence[str]) -> str:
    if not items:
        return "    # (none specified)"
    return "\n".join(f"    {json.dumps(item)}" for item in items)


def _render_test_script(pass_tests: Sequence[str], fail_tests: Sequence[str]) -> str:
    pass_entries = _format_bash_array(pass_tests)
    fail_entries = _format_bash_array(fail_tests)

    return dedent(
        f"""\
        #!/usr/bin/env bash
        set -Eeuo pipefail

        REPO_DIR="/testbed/repo"
        LOG_DIR="/logs"
        VERIFIER_DIR="/logs/verifier"
        LOG_FILE="$LOG_DIR/test_output.log"
        REWARD_FILE="$VERIFIER_DIR/reward.txt"

        mkdir -p "$LOG_DIR" "$VERIFIER_DIR"
        : > "$LOG_FILE"
        : > "$REWARD_FILE"

        exec > >(tee -a "$LOG_FILE") 2>&1

        on_error() {{
            echo "One or more tests failed." >&2
            echo 0 > "$REWARD_FILE"
            exit 1
        }}
        trap on_error ERR

        PASS_TESTS=(
        {pass_entries}
        )

        FAIL_TESTS=(
        {fail_entries}
        )

        log() {{
            echo "[swegym] $*"
        }}

        ensure_dependencies() {{
            log "Installing base Python tooling"
            python3 -m pip install --upgrade pip setuptools wheel

            if [ -f requirements-dev.txt ]; then
                log "Installing requirements-dev.txt"
                python3 -m pip install -r requirements-dev.txt || true
            fi

            if [ -f requirements.txt ]; then
                log "Installing requirements.txt"
                python3 -m pip install -r requirements.txt || true
            fi

            if [ -f pyproject.toml ] || [ -f setup.py ]; then
                log "Installing project in editable mode"
                python3 -m pip install -e . || true
            fi

            python3 -m pip install \"pytest>=8.0.0\" \"pytest-xdist>=3.5.0\" || true
        }}

        run_test_group() {{
            local label=\"$1\"
            shift || true
            local tests=(\"$@\" )

            if [ \"${{#tests[@]}}\" -eq 0 ]; then
                log \"No ${{label}} tests to run\"
                return 0
            fi

            for target in \"${{tests[@]}}\"; do
                if [ -z \"$target\" ] || [[ \"$target\" == \"#\"* ]]; then
                    continue
                fi
                log \"Running ${{label}} test: $target\"
                python3 -m pytest -q \"$target\"
            done
        }}

        cd \"$REPO_DIR\"
        ensure_dependencies

        run_test_group \"pass-to-pass\" \"${{PASS_TESTS[@]}}\"
        run_test_group \"fail-to-pass\" \"${{FAIL_TESTS[@]}}\"

        echo 1 > \"$REWARD_FILE\"
        log \"All configured tests succeeded\"
        """
    )


def _render_test_state_py() -> str:
    return dedent(
        """\
        from pathlib import Path


        def test_reward_file_indicates_success():
            reward_path = Path("/logs/verifier/reward.txt")
            assert reward_path.exists(), "Reward file missing"
            content = reward_path.read_text(encoding="utf-8").strip()
            assert content == "1", f"Tests did not succeed (reward={content!r})"
        """
    )


def _write_tests_config(
    task_dir: Path,
    row: Dict[str, str],
    pass_tests: Sequence[str],
    fail_tests: Sequence[str],
) -> None:
    config = {
        "instance_id": row.get("instance_id"),
        "repo": row.get("repo"),
        "base_commit": row.get("base_commit"),
        "version": row.get("version"),
        "pass_to_pass": list(pass_tests),
        "fail_to_pass": list(fail_tests),
        "patch": row.get("patch"),
        "test_patch_length": len(row.get("test_patch", "")),
    }
    config_path = task_dir / "tests" / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Core pipeline


def _select_rows(ds: Dataset, offset: int, limit: Optional[int]) -> List[int]:
    total = len(ds)
    if offset < 0:
        offset = 0
    if offset >= total:
        return []
    if limit is None:
        end = total
    else:
        end = min(total, offset + limit)
    return list(range(offset, end))


def generate_tasks(args: argparse.Namespace) -> Tuple[Path, Dict[str, object]]:
    dataset = load_dataset(DATASET_NAME, split=args.split)
    limit = args.limit if args.limit and args.limit > 0 else None
    indices = _select_rows(dataset, args.offset, limit)

    temp_root = Path(tempfile.mkdtemp(prefix="swegym_tasks_"))
    produced = 0
    skipped: List[Dict[str, object]] = []

    for row_idx in indices:
        row = dataset[row_idx]
        problem = row.get("problem_statement")
        repo = row.get("repo")
        base_commit = row.get("base_commit")

        if not problem or not repo or not base_commit:
            skipped.append(
                {
                    "index": row_idx,
                    "reason": "missing required field",
                    "instance_id": row.get("instance_id"),
                }
            )
            continue

        pass_tests = _normalize_tests(row.get("PASS_TO_PASS"))
        fail_tests = _normalize_tests(row.get("FAIL_TO_PASS"))

        instruction_content = _render_instruction(row, pass_tests, fail_tests)
        dockerfile_content = _render_dockerfile(row)
        solution_content = _render_solution_script(row.get("test_patch", ""))
        test_sh_content = _render_test_script(pass_tests, fail_tests)
        test_py_content = _render_test_state_py()

        metadata = {
            "source": DATASET_NAME,
            "split": args.split,
            "instance_id": row.get("instance_id"),
            "repo": repo,
            "base_commit": base_commit,
            "version": row.get("version"),
            "pass_to_pass": pass_tests,
            "fail_to_pass": fail_tests,
            "created_at": row.get("created_at"),
        }

        task_dir = create_task_directory_unified(
            output_dir=temp_root,
            task_id=produced,
            instruction_content=instruction_content,
            dataset_prefix=args.dataset_prefix,
            metadata=metadata,
            solution_content=solution_content,
            test_sh_content=test_sh_content,
            test_py_content=test_py_content,
            dockerfile_content=dockerfile_content,
            task_toml_content=TASK_TOML_CONTENT,
        )

        _write_tests_config(task_dir, row, pass_tests, fail_tests)
        produced += 1

    if produced == 0:
        raise RuntimeError("No SWE-Gym tasks were generated; check filters/offsets.")

    artifacts: Dict[str, object] = {
        "dataset": DATASET_NAME,
        "split": args.split,
        "requested_limit": args.limit,
        "applied_limit": limit,
        "offset": args.offset,
        "produced_tasks": produced,
        "skipped": skipped,
    }
    return temp_root, artifacts


# --------------------------------------------------------------------------- #
# CLI entrypoint


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_dir, artifacts = generate_tasks(args)
    final_path = finalize_dataset_output(dataset_dir, args.output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(final_path),
                **artifacts,
            },
            indent=2,
        )
    )

    if args.target_repo and not args.no_upload:
        upload_tasks_to_hf(
            dataset_path=str(final_path),
            repo_id=args.target_repo,
            private=args.hf_private,
            token=args.hf_token,
        )


if __name__ == "__main__":
    main()
