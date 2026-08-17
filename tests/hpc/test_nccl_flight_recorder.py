import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "hpc" / "shell_utils" / "nccl_flight_recorder.sh"


def run_setup(
    experiments_dir: Path, *, environment: dict[str, str] | None = None
) -> dict[str, str]:
    command = f"""
source {shlex.quote(str(SETUP_SCRIPT))}
setup_nccl_flight_recorder {shlex.quote(str(experiments_dir))}
printf 'buffer=%s\n' "$TORCH_NCCL_TRACE_BUFFER_SIZE"
printf 'dump_on_timeout=%s\n' "$TORCH_NCCL_DUMP_ON_TIMEOUT"
printf 'debug_prefix=%s\n' "$TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
printf 'fr_prefix=%s\n' "$TORCH_FR_DUMP_TEMP_FILE"
"""
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    names = {"buffer", "dump_on_timeout", "debug_prefix", "fr_prefix"}
    return dict(
        entry
        for line in result.stdout.splitlines()
        if "=" in line and (entry := line.split("=", 1))[0] in names
    )


def test_setup_creates_job_scoped_durable_dump_destination(tmp_path: Path) -> None:
    experiments_dir = tmp_path / "experiment"

    values = run_setup(experiments_dir, environment={"SLURM_JOB_ID": "1390513"})

    dump_dir = experiments_dir / "nccl_fr" / "1390513"
    expected_prefix = str(dump_dir / "nccl_fr_rank_")
    assert dump_dir.is_dir()
    assert values == {
        "buffer": "20000",
        "dump_on_timeout": "1",
        "debug_prefix": expected_prefix,
        "fr_prefix": expected_prefix,
    }


def test_setup_preserves_explicit_recorder_settings(tmp_path: Path) -> None:
    experiments_dir = tmp_path / "experiment"
    configured_prefix = tmp_path / "custom" / "trace_rank_"

    values = run_setup(
        experiments_dir,
        environment={
            "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": str(configured_prefix),
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "0",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "4096",
        },
    )

    assert configured_prefix.parent.is_dir()
    assert values == {
        "buffer": "4096",
        "dump_on_timeout": "0",
        "debug_prefix": str(configured_prefix),
        "fr_prefix": str(configured_prefix),
    }
