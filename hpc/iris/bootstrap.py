"""Command wrapping for Iris task entrypoints."""

from __future__ import annotations

import os
import shlex


def wrap_task_command(
    command: list[str],
    *,
    extras: list[str],
    needs_tpu_runtime_patch: bool,
) -> list[str]:
    """Wrap Python entrypoints with the OT-Agent Iris runtime bootstrap."""
    if not (command and command[0] == "python" and len(command) >= 2):
        return command

    script_path = command[1]
    script_argv = command[2:]
    py_bootstrap = (
        "import sys; "
        "sys.path.append('/app'); "
        "sys.argv = sys.argv[1:]; "
        "import runpy; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    extras_flags = " ".join(
        f"--extra {shlex.quote(e.split(':', 1)[-1])}" for e in extras
    )
    quiet = "" if os.environ.get("IRIS_DEBUG_UV_RESYNC") else "--quiet"
    # The workspace is mounted at /app for every Iris task, so we still need a
    # frozen sync there even when the task image has a baked environment.  Do
    # not use ``--reinstall``: it needlessly clobbers the image environment on
    # every retry and, more importantly, lets a runtime bootstrap hide a stale
    # image/dependency declaration.  ``--frozen`` makes the checked-in uv.lock
    # the sole dependency authority.
    resync_cmd = (
        "cd /app && "
        f"uv sync {quiet} --frozen --link-mode=copy "
        f"--all-packages --no-group dev {extras_flags}".rstrip()
    )
    patch_cmd = (
        "python scripts/iris/patch_tpu_inference.py"
        if needs_tpu_runtime_patch
        else "true"
    )
    py_invoke = shlex.join(
        ["python", "-c", py_bootstrap, script_path, *script_argv]
    )
    bash_cmd = (
        f"set -e; {resync_cmd}; "
        "export PATH=/app/.venv/bin:$PATH; "
        f"{patch_cmd}; exec {py_invoke}"
    )
    return ["bash", "-c", bash_cmd]
