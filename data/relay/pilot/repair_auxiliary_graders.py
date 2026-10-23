"""Create explicit verifier-only copies for selected Snowball recovery tasks."""

import hashlib
import json
from pathlib import Path
import shutil


TASKS = frozenset(
    {
        "mcmc-sampling-stan",
        "todos-api",
        "pytorch-model-recovery",
        "torch-pipeline-parallelism",
    }
)


def stage_task(source: Path, destination: Path) -> Path:
    """Copy a task and repair verifier execution without changing assertions.

    MCMC and Todos use binary all-tests-pass rewards, so pytest fail-fast
    preserves the reward predicate while avoiding later hangs after a failure.
    PyTorch tasks select the same pinned torch release's CPU wheel instead of
    installing CUDA dependencies into their CPU-only evaluation sandboxes.
    Pipeline verification also stops at the first failed assertion and limits
    BLAS threads to the task's declared one CPU.
    """
    if source.name not in TASKS:
        raise ValueError(f"No auxiliary grader repair for {source.name}")
    source_script = source / "tests" / "test.sh"
    before = source_script.read_text()
    changes = []
    if source.name in {"pytorch-model-recovery", "torch-pipeline-parallelism"}:
        anchor = "uvx \\\n"
        replacement = (
            "uvx \\\n"
            "  --index https://download.pytorch.org/whl/cpu \\\n"
            "  --index-strategy unsafe-best-match \\\n"
        )
        changes.append("cpu_torch_index")
    else:
        anchor = "  pytest --ctrf "
        replacement = "  pytest -x --ctrf "
        changes.append("pytest_fail_fast")
    if before.count(anchor) != 1:
        raise ValueError(f"Unexpected verifier script in {source}")
    after = before.replace(anchor, replacement, 1)
    if source.name == "torch-pipeline-parallelism":
        anchor = "  pytest --ctrf "
        if after.count(anchor) != 1:
            raise ValueError(f"Unexpected pytest invocation in {source}")
        after = after.replace(anchor, "  pytest -x --ctrf ", 1)
        # The task declares one CPU. Avoid each torch subprocess creating a
        # host-sized BLAS thread pool while preserving all numerical checks.
        anchor = "uvx \\\n"
        after = after.replace(
            anchor,
            "export OMP_NUM_THREADS=1\n"
            "export MKL_NUM_THREADS=1\n"
            "export OPENBLAS_NUM_THREADS=1\n\n" + anchor,
            1,
        )
        changes.extend(["pytest_fail_fast", "single_cpu_thread_limits"])
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
        "change": changes[0] if len(changes) == 1 else changes,
    }
    (destination / "auxiliary_grader_repair.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return destination
