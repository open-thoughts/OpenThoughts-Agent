from __future__ import annotations

import argparse
import errno
import os
import pty
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional


def parse_bool_flag(value: Any) -> bool:
    """Best-effort boolean parser for CLI arguments."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got '{value}'")


def coerce_str_bool_none(
    args_dict: dict[str, Any],
    literal_none_keys: set[str],
    bool_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Normalize string CLI values representing booleans or None tokens."""

    bool_keys = set(bool_keys or [])

    for key, value in list(args_dict.items()):
        if not isinstance(value, str):
            continue
        lowered = value.strip().lower()
        if key in bool_keys and lowered in {"true", "false", "1", "0", "yes", "no", "y", "n", "on", "off"}:
            args_dict[key] = parse_bool_flag(lowered)
        elif lowered == "none":
            args_dict[key] = lowered if key in literal_none_keys else None
    return args_dict


def coerce_numeric_cli_values(args_dict: dict[str, Any]) -> dict[str, Any]:
    """Cast well-known numeric CLI arguments when passed as strings."""

    numeric_fields = {
        "adam_beta1": float,
        "adam_beta2": float,
        "learning_rate": float,
        "warmup_ratio": float,
        "weight_decay": float,
        "max_grad_norm": float,
        "num_train_epochs": float,
        "max_steps": int,
        "chunk_size": int,
    }
    for key, caster in numeric_fields.items():
        if key not in args_dict or args_dict[key] is None:
            continue
        value = args_dict[key]
        if isinstance(value, (int, float)):
            continue
        try:
            args_dict[key] = caster(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Expected {key} to be {caster.__name__}-like, got {value!r}") from exc
    return args_dict


def run_harbor_cli(cmd: List[str], log_path: Optional[Path] = None) -> int:
    """Run Harbor CLI with proper TTY handling.

    Harbor CLI requires a pseudo-terminal (PTY) for proper output handling.
    Without it, Harbor may buffer output indefinitely or hang waiting for
    terminal interaction.

    Args:
        cmd: Command list to execute (e.g., ["harbor", "jobs", "start", ...])
        log_path: Optional path to write Harbor output to a file instead of stdout.

    Returns:
        Exit code from Harbor process.

    Raises:
        subprocess.CalledProcessError: If Harbor exits with non-zero status.
    """
    if log_path:
        # File-based output - no PTY needed (line-buffered for real-time tail access)
        with open(log_path, "w", encoding="utf-8", buffering=1) as harbor_log_file:
            print(f"Streaming Harbor output to {log_path}")
            result = subprocess.run(
                cmd,
                check=False,
                stdout=harbor_log_file,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result.returncode

    # PTY-based output for interactive-like behavior
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=False,
        )
        os.close(slave_fd)

        # Read and forward output in real-time
        while True:
            try:
                data = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                break
            if not data:
                break
            os.write(sys.stdout.fileno(), data)
    finally:
        os.close(master_fd)

    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return ret


__all__ = [
    "parse_bool_flag",
    "coerce_str_bool_none",
    "coerce_numeric_cli_values",
    "run_harbor_cli",
]
