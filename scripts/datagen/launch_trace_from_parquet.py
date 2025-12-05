#!/usr/bin/env python3
"""
Convenience script for running trace-only datagen jobs on Oumi.

Workflow:
1. Downloads a sandbox dataset from Hugging Face (expects Parquet layout).
2. Converts the Parquet snapshot into extracted task folders.
3. Launches the requested trace job via `python -m hpc.launch`, wiring
   `--trace_input_path` to the extracted directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


from data.commons import download_hf_dataset  # noqa: E402
from data.generation.utils import load_datagen_config  # noqa: E402
from hpc.datagen_launch_utils import resolve_datagen_config_path  # noqa: E402
from scripts.harbor import tasks_parquet_converter as tpc  # noqa: E402


def _find_parquet_file(dataset_dir: Path, explicit_name: str | None = None) -> Path:
    if explicit_name:
        path = dataset_dir / explicit_name
        if not path.exists():
            raise FileNotFoundError(f"Specified parquet file not found: {path}")
        return path.resolve()

    candidates = sorted(dataset_dir.rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No parquet files found under downloaded dataset: {dataset_dir}"
        )
    if len(candidates) > 1:
        print(
            "Warning: multiple parquet files detected. "
            f"Using the first match: {candidates[0]}"
        )
    return candidates[0].resolve()


def _maybe_clean_directory(path: Path, overwrite: bool) -> bool:
    if path.exists():
        if not overwrite and any(path.iterdir()):
            print(
                f"[trace-launch] Extraction target '{path}' already exists; assuming it is valid."
            )
            return False
        if overwrite:
            for child in path.iterdir():
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
    else:
        path.mkdir(parents=True, exist_ok=True)
    return True


def _normalize_json_arg(name: str, value: str | None) -> str | None:
    if value is None or value == "":
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object (received {type(parsed).__name__})")
    return json.dumps(parsed)


def _build_launch_command(args: argparse.Namespace, trace_input: Path) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "hpc.launch",
        "--job_type",
        "datagen",
        "--experiments_dir",
        str(args.experiments_dir),
        "--enable_task_gen",
        "False",
        "--enable_trace_gen",
        "--datagen_config",
        str(args.datagen_config),
    ]

    if args.datagen_extra_args:
        cmd.extend(["--datagen_extra_args", args.datagen_extra_args])

    if args.pinggy_persistent_url is not None:
        cmd.extend(["--pinggy_persistent_url", args.pinggy_persistent_url])
    if args.pinggy_ssh_command is not None:
        cmd.extend(["--pinggy_ssh_command", args.pinggy_ssh_command])
    if args.pinggy_debugger_url is not None:
        cmd.extend(["--pinggy_debugger_url", args.pinggy_debugger_url])

    cmd.extend(
        [
            "--trace_script",
            args.trace_script,
            "--trace_target_repo",
            args.trace_target_repo,
            "--trace_engine",
            args.trace_engine,
            "--trace_backend",
            args.trace_backend,
            "--trace_input_path",
            str(trace_input),
            "--trace_harbor_config",
            str(args.trace_harbor_config),
            "--gpus_per_node",
            str(args.gpus_per_node),
            "--time_limit",
            args.time_limit,
        ]
    )

    if args.trace_model:
        cmd.extend(["--trace_model", args.trace_model])
    if args.trace_agent_name:
        cmd.extend(["--trace_agent_name", args.trace_agent_name])
    if args.trace_agent_kwargs:
        cmd.extend(["--trace_agent_kwargs", args.trace_agent_kwargs])
    if args.trace_n_concurrent is not None:
        cmd.extend(["--trace_n_concurrent", str(args.trace_n_concurrent)])
    if args.trace_env:
        cmd.extend(["--trace_env", args.trace_env])

    if args.disable_verification:
        cmd.append("--disable_verification")
    if args.trace_agent_timeout_sec is not None:
        cmd.extend(["--trace_agent_timeout_sec", str(args.trace_agent_timeout_sec)])
    if args.sandbox_cpu is not None:
        cmd.extend(["--sandbox_cpu", str(args.sandbox_cpu)])
    if args.sandbox_memory_gb is not None:
        cmd.extend(["--sandbox_memory_gb", str(args.sandbox_memory_gb)])
    if args.sandbox_disk_gb is not None:
        cmd.extend(["--sandbox_disk_gb", str(args.sandbox_disk_gb)])
    if args.sandbox_gpu is not None:
        cmd.extend(["--sandbox_gpu", str(args.sandbox_gpu)])
    if args.trace_max_tokens is not None:
        cmd.extend(["--trace_max_tokens", str(args.trace_max_tokens)])
    if args.chunk_size:
        cmd.extend(["--chunk_size", str(args.chunk_size)])
    if args.additional_launch_args:
        cmd.extend(args.additional_launch_args)
    return cmd


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Parquet tasks, extract them, and launch a trace job."
    )
    parser.add_argument(
        "--tasks_repo",
        default="mlfoundations-dev/nl2bash",
        help="Hugging Face dataset repo ID to download (default: %(default)s)",
    )
    parser.add_argument(
        "--tasks_revision",
        default=None,
        help="Optional revision/commit for the tasks repo.",
    )
    parser.add_argument(
        "--parquet_name",
        default=None,
        help="Specific parquet filename inside the snapshot. If omitted, the first *.parquet found is used.",
    )
    parser.add_argument(
        "--extracted_dir",
        default=None,
        help="Directory to extract tasks into. Defaults to <experiments_dir>/tasks_extracted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing extracted directory content.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Maximum number of tasks per trace chunk when splitting trace jobs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions and launch command without executing.",
    )
    parser.add_argument(
        "--extract_tasks_only",
        action="store_true",
        help=(
            "Extract tasks and exit without launching the trace job. "
            "Useful when you only want to materialize the tasks directory."
        ),
    )

    parser.add_argument(
        "--experiments_dir",
        required=True,
        help="Experiments directory passed through to hpc.launch.",
    )
    parser.add_argument(
        "--datagen_config",
        required=True,
        help="Path to the datagen engine YAML consumed by hpc.launch.",
    )
    parser.add_argument(
        "--pinggy_persistent_url",
        default=None,
        help="Override Pinggy persistent URL forwarded to hpc.launch.",
    )
    parser.add_argument(
        "--pinggy_ssh_command",
        default=None,
        help="Override Pinggy SSH tunnel command forwarded to hpc.launch.",
    )
    parser.add_argument(
        "--pinggy_debugger_url",
        default=None,
        help="Override Pinggy debugger URL forwarded to hpc.launch.",
    )
    parser.add_argument(
        "--datagen_extra_args",
        default=None,
        help="Additional CLI switches forwarded to the datagen generator.",
    )
    parser.add_argument(
        "--trace_script",
        default="data/code_contests/generate_abstract.py",
        help="Trace generation script (default: %(default)s).",
    )
    parser.add_argument(
        "--trace_target_repo",
        required=True,
        help="Target Hugging Face repo for traces.",
    )
    parser.add_argument(
        "--trace_harbor_config",
        required=True,
        help="Harbor job YAML controlling the trace run.",
    )
    parser.add_argument(
        "--trace_model",
        default=None,
        help="Override agent model for trace generation.",
    )
    parser.add_argument(
        "--trace_agent_name",
        default=None,
        help="Override agent name for trace generation.",
    )
    parser.add_argument(
        "--trace_agent_kwargs",
        default=None,
        help="JSON object of additional agent kwargs (e.g. '{\"max_episodes\": 32}').",
    )
    parser.add_argument(
        "--trace_n_concurrent",
        type=int,
        default=None,
        help="Override concurrent trace trials.",
    )
    parser.add_argument(
        "--trace_env",
        default=None,
        help="Override Harbor environment type for trace generation (docker, daytona, etc.).",
    )
    parser.add_argument(
        "--trace_agent_timeout_sec",
        type=float,
        default=None,
        help="Override Harbor agent timeout (seconds) for trace generation.",
    )
    parser.add_argument(
        "--trace_engine",
        default="vllm_local",
        help="Trace engine (default: %(default)s).",
    )
    parser.add_argument(
        "--trace_backend",
        default="vllm",
        help="Trace backend (default: %(default)s).",
    )
    parser.add_argument(
        "--disable_verification",
        action="store_true",
        help="Disable Harbor verification during trace generation.",
    )

    parser.add_argument(
        "--trace_max_tokens",
        type=int,
        default=None,
        help="Maximum output tokens per completion during trace generation.",
    )
    parser.add_argument(
        "--sandbox_cpu",
        type=int,
        default=None,
        help="Override Daytona sandbox vCPU allocation.",
    )
    parser.add_argument(
        "--sandbox_memory_gb",
        type=int,
        default=None,
        help="Override Daytona sandbox memory in GB.",
    )
    parser.add_argument(
        "--sandbox_disk_gb",
        type=int,
        default=None,
        help="Override Daytona sandbox disk in GB.",
    )
    parser.add_argument(
        "--sandbox_gpu",
        type=int,
        default=None,
        help="Override Daytona sandbox GPU allocation.",
    )

    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=8,
        help="GPUs per node for the vLLM server (default: %(default)s).",
    )
    parser.add_argument(
        "--time_limit",
        default="167:00:00",
        help="Overall job time limit (default: %(default)s).",
    )
    parser.add_argument(
        "--additional_launch_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments appended to the hpc.launch command.",
    )
    args = parser.parse_args(argv)
    args.trace_agent_kwargs = _normalize_json_arg(
        "trace_agent_kwargs", args.trace_agent_kwargs
    )
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    experiments_dir = Path(args.experiments_dir).expanduser().resolve()
    experiments_dir.mkdir(parents=True, exist_ok=True)
    args.experiments_dir = str(experiments_dir)

    datagen_config_path = resolve_datagen_config_path(args.datagen_config)
    # Validate the config eagerly so we fail fast before submitting jobs.
    load_datagen_config(datagen_config_path)
    print(f"[trace-launch] Using datagen config: {datagen_config_path}")
    args.datagen_config = str(datagen_config_path)

    print(f"[trace-launch] Downloading dataset snapshot: {args.tasks_repo}")
    dataset_path_str = download_hf_dataset(args.tasks_repo, revision=args.tasks_revision)
    dataset_path = Path(dataset_path_str)
    print(f"[trace-launch] Dataset cached at: {dataset_path}")

    parquet_path = _find_parquet_file(dataset_path, args.parquet_name)
    print(f"[trace-launch] Using parquet file: {parquet_path}")

    extracted_dir = (
        Path(args.extracted_dir).expanduser().resolve()
        if args.extracted_dir
        else experiments_dir / "tasks_extracted"
    )
    print(f"[trace-launch] Extracting tasks to: {extracted_dir}")
    should_extract = _maybe_clean_directory(extracted_dir, args.overwrite)

    if should_extract:
        if not args.dry_run:
            tpc.from_parquet(str(parquet_path), str(extracted_dir), on_exist="overwrite")
        else:
            print("[trace-launch] DRY RUN: skipping parquet extraction")
    else:
        print("[trace-launch] Reusing existing extracted tasks directory.")

    # If requested, stop after extraction and do not launch the trace job.
    if args.extract_tasks_only:
        print("[trace-launch] --extract_tasks_only set; exiting after extraction.")
        return

    cmd = _build_launch_command(args, extracted_dir)
    print("\n[trace-launch] Launch command:")
    print(" ".join(json.dumps(c) if " " in c else c for c in cmd))

    if args.dry_run:
        print("[trace-launch] DRY RUN: command not executed.")
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
