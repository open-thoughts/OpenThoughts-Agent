"""Create explicit verifier-only copies for selected Snowball recovery tasks."""

import hashlib
import json
from pathlib import Path
import shutil


TASKS = frozenset({"mcmc-sampling-stan", "todos-api", "pytorch-model-recovery"})


def stage_task(source: Path, destination: Path) -> Path:
    """Copy a task and repair verifier execution without changing assertions.

    MCMC and Todos use binary all-tests-pass rewards, so pytest fail-fast
    preserves the reward predicate while avoiding later hangs after a failure.
    PyTorch recovery selects the same pinned torch release's CPU wheel instead
    of installing CUDA dependencies into its CPU-only evaluation sandbox.
    """
    if source.name not in TASKS:
        raise ValueError(f"No auxiliary grader repair for {source.name}")
    source_script = source / "tests" / "test.sh"
    before = source_script.read_text()
    if source.name == "pytorch-model-recovery":
        anchor = "uvx \\\n"
        replacement = (
            "uvx \\\n"
            "  --index https://download.pytorch.org/whl/cpu \\\n"
            "  --index-strategy unsafe-best-match \\\n"
        )
    else:
        anchor = "  pytest --ctrf "
        replacement = "  pytest -x --ctrf "
    if before.count(anchor) != 1:
        raise ValueError(f"Unexpected verifier script in {source}")
    after = before.replace(anchor, replacement, 1)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
    (destination / "tests" / "test.sh").write_text(after)
    manifest = {
        "source": str(source),
        "task": source.name,
        "test_assertions_unchanged": True,
        "verifier_budget_unchanged": True,
        "agent_environment_unchanged": True,
        "source_script_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "repaired_script_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "change": "cpu_torch_index"
        if source.name == "pytorch-model-recovery"
        else "pytest_fail_fast",
    }
    (destination / "auxiliary_grader_repair.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return destination
