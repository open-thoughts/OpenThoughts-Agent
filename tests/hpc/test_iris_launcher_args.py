from dataclasses import dataclass
from pathlib import Path

import pytest

from eval.cloud.launch_eval_iris import EvalIrisLauncher
from hpc.iris.bootstrap import wrap_task_command
from hpc.iris.env import _alias_s3_credentials
from hpc.iris_launch_utils import IrisLauncher, ResolvedIrisAccelerator


REPO_ROOT = Path(__file__).resolve().parents[2]
HARBOR_CONFIG = REPO_ROOT / "hpc/harbor_yaml/eval/dcagent_eval_defaults.yaml"
DATAGEN_CONFIG = REPO_ROOT / "hpc/datagen_yaml/extra/qwen3_coder_30b_a3b_vllm_serve_32k.yaml"
HF_DATASET = "DCAgent2/swebench-verified-random-100-folders"


@dataclass(frozen=True)
class FakeTopology:
    name: str
    chip_count: int
    vm_count: int


@dataclass(frozen=True)
class FakeResourceSpec:
    tpu: str | None
    gpu: str | None
    cpu: float
    memory: str
    disk: str

    def to_proto(self):
        return object()


@dataclass(frozen=True)
class FakeCoscheduling:
    group_by: str


class FakeIrisJobApi:
    tpu_topologies = {
        "v6e-4": FakeTopology(name="v6e-4", chip_count=4, vm_count=1),
        "v5p-32": FakeTopology(name="v5p-32", chip_count=16, vm_count=4),
    }

    def get_tpu_topology(self, tpu_type):
        try:
            return self.tpu_topologies[tpu_type]
        except KeyError as exc:
            raise ValueError(f"Unknown TPU type: {tpu_type}") from exc

    def build_tpu_alternatives(self, tpu_arg):
        variants = tuple(v.strip() for v in tpu_arg.split(",") if v.strip())
        if not variants:
            raise ValueError("--tpu must specify at least one TPU variant")
        primary_vm_count = self.get_tpu_topology(variants[0]).vm_count
        for variant in variants[1:]:
            if self.get_tpu_topology(variant).vm_count != primary_vm_count:
                raise ValueError("All TPU alternatives must share the same vm_count")
        return list(variants)

    def parse_gpu_spec(self, spec):
        if spec == "H100x8":
            return "H100", 8
        if spec == "H100":
            return "H100", 1
        raise ValueError(f"Unknown GPU spec: {spec!r}")

    def build_resources(self, tpu, gpu, cpu=0.5, memory="1GB", disk="5GB"):
        return FakeResourceSpec(tpu=tpu, gpu=gpu, cpu=cpu, memory=memory, disk=disk)

    def resolve_multinode_defaults(self, tpu, gpu, replicas):
        if tpu and self.get_tpu_topology(tpu).vm_count > 1:
            return replicas or self.get_tpu_topology(tpu).vm_count, FakeCoscheduling(group_by="tpu-name")
        if gpu and replicas and replicas > 1:
            return replicas, FakeCoscheduling(group_by="leafgroup")
        return replicas or 1, None


class DummyIrisLauncher(IrisLauncher):
    def add_task_specific_args(self, parser):
        pass

    def build_task_command(self, args, remote_output_dir):
        return ["true"]


class CapturingEvalIrisLauncher(EvalIrisLauncher):
    def build_task_command(self, args, remote_output_dir):
        self.remote_output_dir = remote_output_dir
        self.command = super().build_task_command(args, remote_output_dir)
        return self.command


def _parse_dummy_args(*argv):
    launcher = DummyIrisLauncher(REPO_ROOT, iris_api=FakeIrisJobApi())
    parser = launcher.create_argument_parser()
    args = parser.parse_args(list(argv))
    launcher._normalize_accelerator_args(args)
    return args


def _parse_eval_args(*extra):
    argv = [
        "--harbor_config",
        str(HARBOR_CONFIG),
        "--datagen_config",
        str(DATAGEN_CONFIG),
        "--dataset_path",
        HF_DATASET,
        "--gpu",
        "H100x8",
        *extra,
    ]
    launcher = CapturingEvalIrisLauncher(REPO_ROOT, iris_api=FakeIrisJobApi())
    parser = launcher.create_argument_parser()
    return launcher, parser.parse_args(argv)


def _option_value(command, flag):
    return command[command.index(flag) + 1]


def _equals_option_values(command, flag):
    prefix = f"{flag}="
    return [arg[len(prefix):] for arg in command if arg.startswith(prefix)]


def test_gpu_parsing_and_accelerator_resolution_cover_iris_specs():
    gpu_args = _parse_dummy_args("--gpu", "H100x8")
    gpu = gpu_args._resolved_iris_accelerator
    assert (gpu_args.gpu, gpu_args.tpu) == ("H100x8", None)
    assert isinstance(gpu, ResolvedIrisAccelerator)
    assert (gpu.is_gpu, gpu.downstream_eval_device_count, gpu.vm_count) == (True, 8, 1)
    assert (gpu.default_extras, gpu.uses_iris_serve, gpu.needs_tpu_runtime_patch) == (
        ["datagen"],
        False,
        False,
    )

    tpu = _parse_dummy_args("--tpu", "v5p-32")._resolved_iris_accelerator
    assert (tpu.is_tpu, tpu.primary_tpu, tpu.downstream_eval_device_count, tpu.vm_count) == (
        True,
        "v5p-32",
        16,
        4,
    )
    assert (tpu.default_extras, tpu.uses_iris_serve, tpu.needs_tpu_runtime_patch) == (
        ["datagen-tpu"],
        True,
        True,
    )

    default_tpu = _parse_dummy_args()._resolved_iris_accelerator
    assert (default_tpu.is_tpu, default_tpu.primary_tpu) == (True, DummyIrisLauncher.default_tpu)

    with pytest.raises(SystemExit, match="mutually exclusive"):
        _parse_dummy_args("--gpu", "H100x8", "--tpu", "v5p-32")


def test_eval_iris_gpu_s3_dry_run_covers_runtime_paths():
    launcher, args = _parse_eval_args(
        "--output-mode",
        "s3",
        "--s3-output-dir",
        "s3://ota-r2/evals",
        "--job_name",
        "gpu-infra-smoke",
        "--dry-run",
    )
    assert launcher.run(args) == 0

    expected_remote_output = "s3://ota-r2/evals/gpu-infra-smoke"
    expected_work_output = "/tmp/ot-agent-runs/gpu-infra-smoke"
    assert launcher.remote_output_dir == expected_remote_output
    assert args._work_output_dir == expected_work_output
    assert _option_value(launcher.command, "--experiments_dir") == expected_work_output
    assert "--jobs-dir=s3://ota-r2/evals" in _equals_option_values(launcher.command, "--harbor_extra_arg")
    assert (args.dataset_path, args.gpus) == (HF_DATASET, 8)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--output-mode", "s3", "--s3-output-dir", ""), "--s3-output-dir is required"),
        (("--output-mode", "gcs"), "must not write to GCS"),
        (
            ("--output-mode", "s3", "--s3-output-dir", "s3://ota-r2/evals", "--upload_to_database"),
            "--upload_to_database",
        ),
        (
            ("--output-mode", "s3", "--s3-output-dir", "s3://ota-r2/evals", "--replicas", "2"),
            "GPU eval replicas > 1 need task sharding",
        ),
    ],
)
def test_eval_iris_gpu_rejects_unsafe_output_and_replica_modes(extra, message):
    launcher, args = _parse_eval_args("--dry_run", *extra)
    with pytest.raises(SystemExit, match=message):
        launcher.run(args)


def test_s3_credential_aliasing_keeps_tpu_marin_precedence_and_gpu_r2():
    tpu_env = {
        "AWS_ACCESS_KEY_ID": "shell",
        "AWS_SECRET_ACCESS_KEY": "shell-secret",
        "AWS_SESSION_TOKEN": "stale",
        "MARIN_HMAC_ACCESS_ID": "marin",
        "MARIN_HMAC_SECRET": "marin-secret",
    }
    _alias_s3_credentials(tpu_env, ResolvedIrisAccelerator(kind="tpu"))
    assert (
        tpu_env["AWS_ENDPOINT_URL"],
        tpu_env["AWS_ACCESS_KEY_ID"],
        tpu_env["AWS_SECRET_ACCESS_KEY"],
        "AWS_SESSION_TOKEN" in tpu_env,
    ) == ("https://storage.googleapis.com", "marin", "marin-secret", False)

    gpu_env = {
        "R2_ENDPOINT": "r2.example.com",
        "R2_ACCESS_KEY_ID": "r2",
        "R2_SECRET_ACCESS_KEY": "r2-secret",
    }
    _alias_s3_credentials(gpu_env, ResolvedIrisAccelerator(kind="gpu"))
    assert (
        gpu_env["AWS_ENDPOINT_URL"],
        gpu_env["AWS_ACCESS_KEY_ID"],
        gpu_env["AWS_SECRET_ACCESS_KEY"],
    ) == ("https://r2.example.com", "r2", "r2-secret")


def test_iris_bootstrap_uses_synced_venv_and_gates_tpu_patch():
    gpu_script = wrap_task_command(
        ["python", "eval/local/run_eval.py", "--flag", "value with spaces"],
        extras=["datagen"],
        needs_tpu_runtime_patch=False,
    )[2]
    assert "uv sync" in gpu_script
    assert "--extra datagen" in gpu_script
    assert "export PATH=/app/.venv/bin:$PATH" in gpu_script
    assert "patch_tpu_inference.py" not in gpu_script

    tpu_script = wrap_task_command(
        ["python", "eval/local/run_eval.py"],
        extras=["datagen-tpu"],
        needs_tpu_runtime_patch=True,
    )[2]
    assert "python scripts/iris/patch_tpu_inference.py" in tpu_script
