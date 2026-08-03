"""Docker-based oracle gate: validate that gold→1 AND empty→0.

Generalized from data/nemotron_cpp_v2/generate.py:validate_task().

Usage:
    from data.rl_converters.oracle_gate import OracleGate

    gate = OracleGate(image="my-rl-image:v1")
    ok, reason = gate.validate(task_dir="/tmp/task_0")
    # ok=True iff gold→1 AND empty→0
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class OracleGate:
    """Two-sided Docker oracle gate.

    For each task, runs the verifier (tests/test.sh) in a Docker container
    under two scenarios:
    1. Gold: /app populated with solution/ contents → reward must be 1
    2. Empty: /app empty → reward must be 0

    A task passes iff both conditions hold.
    """

    def __init__(self, image: str, timeout: int = 300):
        self.image = image
        self.timeout = timeout

    def _run_verifier(
        self,
        task_dir: str | Path,
        populate_gold: bool,
    ) -> int:
        """Run tests/test.sh in Docker, return reward (0 or 1)."""
        task_dir = Path(task_dir)
        with tempfile.TemporaryDirectory() as workdir:
            wd = Path(workdir)
            app_dir = wd / "app"
            tests_mount = wd / "tests_mount"
            logs_dir = wd / "logs" / "verifier"

            app_dir.mkdir(parents=True)
            tests_mount.mkdir(parents=True)
            logs_dir.mkdir(parents=True)

            # Copy the test files into the mount
            src_tests = task_dir / "tests"
            if src_tests.exists():
                for f in src_tests.iterdir():
                    shutil.copy2(f, tests_mount / f.name)

            # Populate /app with gold solution if requested
            if populate_gold:
                solution_dir = task_dir / "solution"
                if solution_dir.exists():
                    for f in solution_dir.iterdir():
                        if f.name == "solve.sh":
                            continue
                        shutil.copy2(f, app_dir / f.name)

            cmd = (
                "mkdir -p /logs/verifier && "
                "cp -r /work/tests_mount/* /tests/ 2>/dev/null; "
                "cd /work/app && "
                "bash /tests/test.sh > /tmp/verifier_stdout.txt 2>&1; "
                "cat /logs/verifier/reward.txt 2>/dev/null || echo 'NO_REWARD'"
            )

            try:
                r = subprocess.run(
                    [
                        "docker", "run", "--rm",
                        "-v", f"{wd}:/work",
                        "-v", f"{tests_mount}:/tests",
                        "-v", f"{wd}/logs:/logs",
                        self.image,
                        "bash", "-c", cmd,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                output = (r.stdout or "").strip()
                if "1" in output.split("\n")[-1] if output else False:
                    return 1
                # More robust check
                reward_line = [l for l in output.split("\n") if l.strip() in ("0", "1")]  # noqa: E741
                if reward_line:
                    return int(reward_line[-1].strip())
                return 0
            except subprocess.TimeoutExpired:
                return 0
            except Exception:
                return 0

    def validate(self, task_dir: str | Path) -> tuple[bool, str]:
        """Return (passed, reason). Passes iff gold→1 AND empty→0."""
        gold_reward = self._run_verifier(task_dir, populate_gold=True)
        if gold_reward != 1:
            return False, "gold_not_1"

        empty_reward = self._run_verifier(task_dir, populate_gold=False)
        if empty_reward != 0:
            return False, "empty_not_0"

        return True, "ok"

    def validate_batch(
        self,
        task_dirs: list[str | Path],
        workers: int = 4,
    ) -> list[tuple[Path, bool, str]]:
        """Validate multiple tasks in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(self.validate, td): td for td in task_dirs
            }
            for fut in as_completed(futures):
                td = futures[fut]
                try:
                    ok, reason = fut.result()
                except Exception as e:
                    ok, reason = False, f"exception:{e}"
                results.append((Path(td), ok, reason))
        return results
