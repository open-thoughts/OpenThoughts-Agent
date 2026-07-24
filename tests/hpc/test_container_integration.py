"""Tests for the container runtime (Pyxis/Enroot) integration on the HPC model.

These tests guard the flag-off byte-identical invariant (G1): when container_image
is unset, behavior is unchanged for all existing clusters. They also verify the
container-aware code paths produce correct output for containerized clusters.
"""

import pytest
from hpc.hpc import HPC, clusters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plain_cluster():
    """A minimal non-container cluster for flag-off testing."""
    return HPC(
        name="test_plain",
        hostname_pattern=r"plain\.example\.com",
        dotenv_filename="plain.env",
        account="testacct",
        partition="gpu",
        gpus_per_node=8,
        cpus_per_node=128,
        internet_node=True,
        gpus_type="H100",
        total_partition_nodes=10,
    )


@pytest.fixture
def container_cluster():
    """A containerized cluster mimicking EmpireAI's config."""
    return HPC(
        name="test_container",
        hostname_pattern=r"ctr\.example\.com",
        dotenv_filename="container.env",
        account="testacct",
        partition="gpu",
        gpus_per_node=4,
        cpus_per_node=144,
        internet_node=True,
        gpus_type="B200",
        total_partition_nodes=72,
        container_image="/fake/path/to/image.sqsh",
        container_mount_home=True,
        container_remap_root=False,
        env_vars={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        nccl_settings={"NCCL_SOCKET_IFNAME": "bond0"},
    )


# ---------------------------------------------------------------------------
# Stage 1: HPC container fields
# ---------------------------------------------------------------------------


class TestContainerFields:
    """Verify the container fields exist and have correct defaults."""

    def test_flag_off_all_existing_clusters_have_no_container(self):
        """G1 invariant: every registered cluster has container_image=None."""
        containerized = [c.name for c in clusters if c.container_image is not None]
        # EmpireAI is the ONLY containerized cluster (added by this change).
        assert containerized == ["empireai"], (
            f"Unexpected containerized clusters: {containerized}"
        )

    def test_container_defaults_are_none(self, plain_cluster):
        """A cluster without container_image set has all container defaults off."""
        assert plain_cluster.container_image is None
        assert plain_cluster.is_containerized is False
        assert plain_cluster.slurm_segment == 0

    def test_container_cluster_is_containerized(self, container_cluster):
        """A cluster with container_image set is recognized as containerized."""
        assert container_cluster.container_image == "/fake/path/to/image.sqsh"
        assert container_cluster.is_containerized is True

    def test_slurm_segment_default(self, plain_cluster):
        """slurm_segment defaults to 0 (no --segment directive)."""
        assert plain_cluster.slurm_segment == 0


# ---------------------------------------------------------------------------
# Stage 1: get_container_srun_flags
# ---------------------------------------------------------------------------


class TestContainerSrunFlags:
    """Verify the srun container flag generation."""

    def test_no_container_returns_empty(self, plain_cluster):
        """Flag-off: no container_image → empty srun flags."""
        assert plain_cluster.get_container_srun_flags() == ""

    def test_container_image_in_flags(self, container_cluster):
        """Container image appears in the srun flags."""
        flags = container_cluster.get_container_srun_flags()
        assert "--container-image=/fake/path/to/image.sqsh" in flags

    def test_container_mount_home_in_flags(self, container_cluster):
        """--container-mount-home is included by default."""
        flags = container_cluster.get_container_srun_flags()
        assert "--container-mount-home" in flags

    def test_container_remap_root_off_by_default(self, container_cluster):
        """--container-remap-root is NOT included when False."""
        flags = container_cluster.get_container_srun_flags()
        assert "--container-remap-root" not in flags

    def test_container_remap_root_when_set(self):
        """--container-remap-root IS included when True."""
        c = HPC(
            name="test",
            hostname_pattern="t",
            dotenv_filename="t.env",
            account="a",
            partition="p",
            gpus_per_node=1,
            cpus_per_node=1,
            internet_node=True,
            gpus_type="T",
            total_partition_nodes=1,
            container_image="/img.sqsh",
            container_remap_root=True,
        )
        assert "--container-remap-root" in c.get_container_srun_flags()

    def test_container_extra_args(self):
        """container_extra_args is appended to the flags."""
        c = HPC(
            name="test",
            hostname_pattern="t",
            dotenv_filename="t.env",
            account="a",
            partition="p",
            gpus_per_node=1,
            cpus_per_node=1,
            internet_node=True,
            gpus_type="T",
            total_partition_nodes=1,
            container_image="/img.sqsh",
            container_extra_args="--container-mounts=/data:/data",
        )
        flags = c.get_container_srun_flags()
        assert "--container-mounts=/data:/data" in flags


# ---------------------------------------------------------------------------
# Stage 2: get_sbatch_directives with segment
# ---------------------------------------------------------------------------


class TestSegmentDirective:
    """Verify the --segment SBATCH directive."""

    def test_no_segment_when_zero(self, plain_cluster):
        """Flag-off: slurm_segment=0 → no --segment directive."""
        directives = plain_cluster.get_sbatch_directives()
        assert "--segment" not in directives

    def test_segment_emitted_when_set(self):
        """slurm_segment>0 → #SBATCH --segment=N."""
        c = HPC(
            name="test",
            hostname_pattern="t",
            dotenv_filename="t.env",
            account="a",
            partition="p",
            gpus_per_node=4,
            cpus_per_node=144,
            internet_node=True,
            gpus_type="B200",
            total_partition_nodes=72,
            slurm_segment=2,
        )
        directives = c.get_sbatch_directives()
        assert "#SBATCH --segment=2" in directives


# ---------------------------------------------------------------------------
# Stage 2: resolve_conda_activate container-aware
# ---------------------------------------------------------------------------


class TestResolveCondaActivate:
    """Verify conda activation is skipped for containerized clusters."""

    def test_conda_for_plain_cluster(self, plain_cluster):
        """Non-container cluster gets its conda_activate string."""
        from hpc.launch_utils import resolve_conda_activate

        plain_cluster.conda_activate = (
            "source /opt/conda/etc/profile.d/conda.sh && conda activate myenv"
        )
        result = resolve_conda_activate(plain_cluster, {})
        assert "conda activate myenv" in result

    def test_conda_empty_for_container_cluster(self, container_cluster):
        """Container cluster gets empty string (env is in the .sqsh)."""
        from hpc.launch_utils import resolve_conda_activate

        result = resolve_conda_activate(container_cluster, {})
        assert result == ""


# ---------------------------------------------------------------------------
# Stage 3: EmpireAI HPC config
# ---------------------------------------------------------------------------


class TestEmpireAIConfig:
    """Verify the EmpireAI cluster registration is complete and correct."""

    @pytest.fixture
    def empireai(self):
        for c in clusters:
            if c.name == "empireai":
                return c
        pytest.fail("empireai not found in clusters list")

    def test_empireai_in_clusters_list(self, empireai):
        """EmpireAI is registered in the clusters list."""
        assert empireai.name == "empireai"

    def test_empireai_container_image(self, empireai):
        """EmpireAI has a container image set."""
        assert empireai.container_image is not None
        assert empireai.container_image.endswith(".sqsh")
        assert empireai.is_containerized is True

    def test_empireai_hostname_pattern(self, empireai):
        """EmpireAI hostname pattern matches Bright cluster node names."""
        import re

        pattern = re.compile(empireai.hostname_pattern)
        assert pattern.match("b1-11-s1-dgx-01-c01")
        assert pattern.match("b6-21-s1-mgmt-01")

    def test_empireai_gpu_directive(self, empireai):
        """EmpireAI GPU directive uses --gres=gpu:b200:N."""
        directive = empireai.get_gpu_directive(4)
        assert "b200" in directive
        assert "4" in directive

    def test_empireai_segment(self, empireai):
        """EmpireAI has --segment=2 for NVLink placement."""
        assert empireai.slurm_segment == 2
        directives = empireai.get_sbatch_directives()
        assert "--segment=2" in directives

    def test_empireai_qos(self, empireai):
        """EmpireAI QoS is set to standard."""
        assert empireai.qos == "standard"

    def test_empireai_nccl_bond0(self, empireai):
        """EmpireAI NCCL settings pin bond0."""
        assert empireai.nccl_settings.get("NCCL_SOCKET_IFNAME") == "bond0"
        nccl_exports = empireai.get_nccl_exports()
        assert "bond0" in nccl_exports

    def test_empireai_env_vars_path_sanitize(self, empireai):
        """EmpireAI env_vars include PATH sanitization."""
        assert "PATH" in empireai.env_vars
        assert "/usr/bin" in empireai.env_vars["PATH"]

    def test_empireai_srun_container_flags(self, empireai):
        """EmpireAI srun flags include container-image + mount-home."""
        flags = empireai.get_container_srun_flags()
        assert "--container-image=" in flags
        assert "--container-mount-home" in flags

    def test_empireai_eval_cluster_view(self, empireai):
        """EmpireAI eval_cluster_view is populated."""
        assert empireai.eval_cluster_view is not None
        assert empireai.eval_cluster_view["cluster_name"] == "empireai"
