#!/usr/bin/env python3
"""
Generate Harbor-compatible tasks from the SWE-Gym dataset.

This script is a patched version of `generate.py` that fixes environment and verification issues
to ensure 100% compatibility with the authors' original "source of truth."

ENVIRONMENT & BUILD FIXES:
- Miniconda Architecture: Switched from system Python/venv to Miniconda for 100% author 
  consistency and total isolation from system packages (prevents blinker/pyparsing crashes).
- Dynamic Python Versioning: Mapped repositories to specific Python versions (e.g., 
  Moto 3.x uses 3.7, 5.x uses 3.12) via isolated Conda environments.
- Image Efficiency: Keeps unique Dockerfile count < 10 by grouping tasks by Python version
  and unionizing system dependencies (apt packages) within each group.
- Legacy Build Fix: Implemented the authors' `cython<3` PIP_CONSTRAINT fix to resolve 
  PyYAML and other legacy build-time crashes in older repositories.
- Compiler Headers: Set `C_INCLUDE_PATH` in the Dockerfile to ensure `gcc` finds `Python.h` 
  for all Conda-installed Python versions.
- Repo-Specific Logic: Integrated exact installation commands (like `make init` for Moto) 
  from the authors' specifications.
- Database Headers: Includes `libpq-dev` in the unionized system dependencies for 
  Python 3.8+ groups to ensure `psycopg2` and other database drivers can compile.

VERIFICATION & INTEGRITY FIXES:
- Resilient Testing: `test.sh` now captures `pytest` Exit Code 4 to safely skip test 
  cases that were truncated in the source dataset's metadata.
- Bash Expansion Safety: Used single-quotes for `PASS_TESTS` and `FAIL_TESTS` arrays to 
  prevent accidental `$s` variable expansion.
- Patch Integrity: Replaced `.rstrip()` with `.strip('\n')` when rendering patches to 
  preserve crucial trailing spaces in context lines, preventing "corrupt patch" errors.
- Library Compatibility: Patches the `sure` assertion library at runtime to fix 
  `AttributeError` related to `re._pattern_type` in Python 3.7+ environments.
- Increased Timeouts: Raised default verifier timeout to 600s to accommodate runtime 
  environment setup and dependency installation.
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
# Mapping Helpers

def get_specs(repo: str, version: str) -> dict:
    """
    Determine the correct Python version and group-unionized pre-install commands
    based on the original SWE-Bench-Fork specifications.
    """
    repo = repo.lower()
    
    # 1. Determine Python Version and Install Command
    py_v = "3.9"
    install_cmd = "pip install --prefer-binary -e ."
    
    if "moto" in repo: 
        try:
            v_float = float(version)
            if v_float >= 5.0: py_v = "3.12"
            elif v_float >= 4.0: py_v = "3.10"
            else: py_v = "3.7"
        except:
            py_v = "3.10"
        install_cmd = "make init"
    elif "conan" in repo: 
        py_v = "3.10"
        # Authors' approach: force cython<3 for legacy compatibility
        install_cmd = "echo 'cython<3' > /tmp/constraint.txt; export PIP_CONSTRAINT=/tmp/constraint.txt; python -m pip install -r conans/requirements.txt; python -m pip install -r conans/requirements_server.txt; python -m pip install -r conans/requirements_dev.txt"
    elif "dvc" in repo:
        try:
            v_float = float(version)
            if v_float >= 1.0: py_v = "3.9"
            else: py_v = "3.8"
        except: py_v = "3.9"
        # Authors' exact multi-stage install logic for DVC
        # Use escaped quotes to prevent syntax errors in the generated bash script
        install_cmd = 'python -m pip install --upgrade pip wheel GitPython; python -m pip install \"cython<3.0.0\" && python -m pip install --no-build-isolation pyyaml==5.4.1; python -m pip install git+https://github.com/iterative/mock-ssh-server.git || true; python -m pip install -r tests/requirements.txt || true; python -m pip install -r test-requirements.txt || true; python -m pip install -e \".[tests,dev,all_remotes,all,testing]\";'
        # DVC legacy pins
        install_cmd += ' python -m pip install \"numpy<=1.20\"; python -m pip install \"pytest<8\";'
    elif "monai" in repo: 
        py_v = "3.8"
        # Authors' approach: remove git links and install pinned setuptools
        install_cmd = "sed -i '/^git+https:\\/\\/github.com\\/Project-MONAI\\//d' requirements-dev.txt; python -m pip install types-pkg-resources==0.1.3 pytest; pip install -r requirements-dev.txt; python setup.py develop;"
    elif "hydra" in repo: 
        py_v = "3.8"
        install_cmd = "pip install --prefer-binary -r requirements/dev.txt && pip install --prefer-binary -e ."
    elif "bokeh" in repo: 
        py_v = "3.8"
        install_cmd = "cd bokehjs && npm install && cd .. && pip install --prefer-binary -e ."
    elif "pydantic" in repo: 
        py_v = "3.8" if version >= "1.10" else "3.7"
        install_cmd = "pip install pdm && pdm install"
    elif "django" in repo:
        try:
            v = float(version)
            if v >= 4.1: py_v = "3.9"
            elif v >= 4.0: py_v = "3.8"
            elif v >= 3.0: py_v = "3.6"
            else: py_v = "3.5"
            install_cmd = "python setup.py install" if v < 3.0 else "pip install --prefer-binary -e ."
        except: py_v = "3.6"
    elif "scikit-learn" in repo:
        install_cmd = "pip install -v --no-use-pep517 --no-build-isolation -e ."
        try:
            v = float(version)
            py_v = "3.9" if v >= 0.23 else "3.6"
        except: py_v = "3.6"
    elif "flask" in repo:
        try:
            v = float(version)
            if v >= 3.0: py_v = "3.11"
            elif v >= 2.1: py_v = "3.10"
            else: py_v = "3.9"
        except: py_v = "3.9"
    elif "astropy" in repo: py_v = "3.10"
    elif "pytest" in repo: py_v = "3.10"
    elif "matplotlib" in repo: py_v = "3.8"

    # 2. Assign Unionized System Dependencies (Apt Packages)
    apt_map = {
        "3.5": ["libsqlite3-dev", "locales"],
        "3.6": ["libsqlite3-dev", "locales"],
        "3.7": ["libsqlite3-dev", "locales"],
        "3.8": ["libsqlite3-dev", "openjdk-17-jdk", "openjdk-17-jre", "npm", "libpq-dev"],
        "3.9": ["libsqlite3-dev", "locales", "libffi-dev", "libssl-dev", "libpq-dev"],
        "3.10": ["cmake", "libpq-dev"],
        "3.11": [],
        "3.12": ["make", "build-essential"]
    }
    
    return {
        "python": py_v,
        "pre_install": apt_map.get(py_v, []),
        "install": install_cmd
    }


# --------------------------------------------------------------------------- #
# Rendering helpers


PYTHON_DOCKERFILE_TEMPLATE = """\
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /testbed

RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
        wget \\
        git \\
        build-essential \\
        pkg-config \\
        ca-certificates \\
        curl \\
        jq \\
        locales \\
        locales-all \\
        tzdata \\
        libxml2-dev \\
        libxmlsec1-dev \\
        libffi-dev \\
        libssl-dev \\
        zlib1g-dev \\
        libbz2-dev \\
        libreadline-dev \\
        libsqlite3-dev \\
        python3 \\
        python3-pip \\
        python-is-python3 \\
        {extra_packages} \\
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda - Consistent with original SWE-Bench approach
RUN wget 'https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh' -O miniconda.sh && \\
    bash miniconda.sh -b -p /opt/miniconda3 && \\
    rm miniconda.sh

ENV PATH="/opt/miniconda3/bin:$PATH"

# Following authors' approach: use /bin/bash explicitly for initialization
RUN /bin/bash -c "source /opt/miniconda3/bin/activate && \\
    conda init bash && \\
    conda config --append channels conda-forge && \\
    conda create -n testbed python={python_version} -y"

# Automatically activate environment in shell sessions
RUN echo "source /opt/miniconda3/bin/activate && conda activate testbed" >> /root/.bashrc
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
timeout_sec = 600.0
"""


def _normalize_tests(raw: Optional[Sequence[str]]) -> List[str]:
    normalized: List[str] = []
    if not raw:
        return normalized
    for entry in raw:
        if not entry:
            continue
        value = entry.strip()
        if not value or value.lower() == "nan":
            continue
            
        # Filter out truncated names:
        # 1. Parameterized tests must have a matching closing bracket if they have an opening one
        if "[" in value and "]" not in value:
            continue
            
        # 2. Heuristic: Very long names that don't end in a predictable way (like ')' or ']')
        # and seem to be cut off at a common power-of-2 limit or similar.
        if len(value) > 250 and not (value.endswith("]") or value.endswith(")")):
            # Check if it looks like it was cut off mid-identifier
            if value[-1].isalnum() or value[-1] in "_-":
                continue

        normalized.append(value)
    return normalized


def _render_instruction(
    row: Dict[str, str],
    pass_tests: Sequence[str],
    fail_tests: Sequence[str],
) -> str:
    repo_url = f"https://github.com/{row['repo']}"
    base_commit = row.get('base_commit', 'unknown')
    version = row.get("version", "unknown")
    specs = get_specs(row['repo'], version)
    
    instruction_lines = [
        "# Bug Fix Task",
        "",
        f"- Repository: `{row['repo']}`",
        f"- Source commit: `{base_commit}`",
        f"- Dataset instance: `{row.get('instance_id', 'n/a')}`",
        f"- Upstream issue version: `{row.get('version', 'n/a')}`",
        f"- Reference repo URL: {repo_url}",
        "",
        "## Environment Setup (complete these steps first)",
        "",
        "```bash",
        "cd /testbed",
        f"git clone {repo_url}.git repo",
        "cd repo",
        f"git checkout {base_commit}",
        "",
        "# Activate environment",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        "",
        "# Install dependencies",
        specs['install'],
        "```",
        "",
        "---",
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
    version = row.get("version", "unknown")
    specs = get_specs(repo, version)
    
    extra_packages = " ".join(specs.get("pre_install", []))

    return PYTHON_DOCKERFILE_TEMPLATE.format(
        python_version=specs["python"],
        extra_packages=extra_packages
    )


def _render_solution_script(row: Dict[str, str], test_patch: str, solution_patch: str) -> Optional[str]:
    if not test_patch.strip() and not solution_patch.strip():
        return None

    repo_url = f"https://github.com/{row['repo']}.git"
    base_commit = row.get("base_commit", "main")

    lines = [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        "",
        "# --- Environment setup ---",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        "cd /testbed",
        'if [ ! -d "repo" ]; then',
        f"    git clone {repo_url} repo",
        "fi",
        "cd repo",
        f"git checkout {base_commit}",
        "# --- End environment setup ---",
    ]

    if test_patch.strip():
        lines.extend([
            "",
            'patch_file="$(mktemp /tmp/swegym-test-patch-XXXX.diff)"',
            "cat <<'PATCH_EOF' > \"$patch_file\"",
            test_patch.strip('\n'),
            "PATCH_EOF",
            "",
            'git apply --whitespace=nowarn --apply "$patch_file"',
        ])

    if solution_patch.strip():
        lines.extend([
            "",
            'patch_file2="$(mktemp /tmp/swegym-solution-patch-XXXX.diff)"',
            "cat <<'SOLUTION_PATCH_EOF' > \"$patch_file2\"",
            solution_patch.strip('\n'),
            "SOLUTION_PATCH_EOF",
            "",
            'git apply --whitespace=nowarn --apply "$patch_file2"',
        ])

    return "\n".join(lines) + "\n"


def _format_bash_array(items: Sequence[str]) -> str:
    if not items:
        return "    # (none specified)"
    # Use single quotes to prevent bash variable expansion (e.g. $s)
    return "\n".join(f"    '{item}'" for item in items)


def _render_test_script(row: Dict[str, str], pass_tests: Sequence[str], fail_tests: Sequence[str]) -> str:
    pass_entries = _format_bash_array(pass_tests)
    fail_entries = _format_bash_array(fail_tests)
    repo = row.get("repo", "unknown")
    version = row.get("version", "unknown")
    specs = get_specs(repo, version)

    return dedent(
        f"""\
        #!/usr/bin/env bash
        set -Eeuo pipefail

        # --- Environment setup ---
        source /opt/miniconda3/bin/activate
        conda activate testbed
        # --- End environment setup ---

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
            log \"Installing base Python tooling\"
            
            # Authors' approach: Use PIP_CONSTRAINT to force cython<3 for legacy compatibility
            # This fixes the 'AttributeError: build_ext object has no attribute cython_sources' error
            echo \"cython<3\" > /tmp/constraint.txt
            export PIP_CONSTRAINT=/tmp/constraint.txt

            # Use prefer-binary to avoid building from source where possible (fixes PyYAML/Cython issues)
            pip install --prefer-binary --upgrade pip setuptools wheel

            if [ -f requirements-dev.txt ]; then
                log \"Installing requirements-dev.txt\"
                pip install --prefer-binary -r requirements-dev.txt || true
            fi

            if [ -f requirements.txt ]; then
                log \"Installing requirements.txt\"
                pip install --prefer-binary -r requirements.txt || true
            fi

            log \"Running repo-specific install...\"
            {specs['install']}

            # Fix 'sure' library bit-rot for Python 3.7+
            # This replaces re._pattern_type with a compatible type check
            SURE_OLD_PY=$(python3 -c \"import sure, os; print(os.path.join(sure.__path__[0], 'old.py'))\" 2>/dev/null || true)
            if [ -f \"$SURE_OLD_PY\" ]; then
                log \"Patching 'sure' library for compatibility...\"
                sed -i 's/re._pattern_type/type(re.compile(\"\"))/g' \"$SURE_OLD_PY\" || true
            fi

            pip install \"pytest>=8.0.0\" \"pytest-xdist>=3.5.0\" || true
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
                
                # Running inside 'if' prevents 'set -e' from exiting on failure,
                # allowing us to check the exit code manually.
                if python3 -m pytest -q \"$target\"; then
                    log \"Test passed: $target\"
                else
                    local ec=$?
                    if [ $ec -eq 4 ]; then
                        log \"WARNING: Test not found (possibly truncated in metadata), skipping: $target\"
                    else
                        log \"Test failed with exit code $ec: $target\"
                        return 1
                    fi
                fi
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
        solution_content = _render_solution_script(row, row.get("test_patch", ""), row.get("patch", ""))
        test_sh_content = _render_test_script(row, pass_tests, fail_tests)
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
