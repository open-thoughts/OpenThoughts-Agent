import pytest

from hpc.hpc import HPC, clusters


@pytest.mark.parametrize(
    "cluster",
    [cluster for cluster in clusters if "NCCL_IB_TIMEOUT" in cluster.nccl_settings],
    ids=lambda cluster: cluster.name,
)
def test_cluster_nccl_ib_timeout_remains_finite(cluster: HPC) -> None:
    timeout = cluster.nccl_settings["NCCL_IB_TIMEOUT"]

    assert 1 <= int(timeout) <= 31, (
        f"{cluster.name} sets NCCL_IB_TIMEOUT={timeout}; NCCL treats values outside 1..31 as infinite"
    )
