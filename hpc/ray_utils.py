"""Ray cluster management for SLURM-based HPC systems.

This module provides a context manager for managing Ray cluster lifecycle
within SLURM jobs, eliminating duplicated Ray setup code across SBATCH scripts.

Usage:
    from hpc.ray_utils import RayCluster, RayClusterConfig

    config = RayClusterConfig(
        num_nodes=4,
        gpus_per_node=4,
        cpus_per_node=48,
    )

    with RayCluster.from_slurm(config) as ray_cluster:
        print(f"Ray cluster ready at {ray_cluster.address}")
        print(f"Total GPUs: {ray_cluster.total_gpus}")
        # ... launch vLLM or other Ray-based workloads
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hpc.hpc import HPC


@dataclass
class RayClusterConfig:
    """Configuration for a Ray cluster on SLURM."""

    num_nodes: int
    gpus_per_node: int
    cpus_per_node: int
    ray_port: int = 6379
    srun_export_env: str = "ALL"
    ray_env_vars: str = ""  # Space-separated KEY=value pairs for Ray workers
    wait_for_cluster_script: str = "scripts/ray/wait_for_cluster.py"
    poll_interval: int = 10
    startup_timeout: int = 600


@dataclass
class RayCluster:
    """Context manager for Ray cluster lifecycle on SLURM.

    This class handles:
    - Starting Ray head node
    - Starting Ray worker nodes
    - Waiting for cluster to be ready
    - Graceful shutdown on exit
    """

    config: RayClusterConfig
    head_ip: str
    node_list: List[str]
    _ray_pids: List[int] = field(default_factory=list)
    _ray_procs: List[subprocess.Popen] = field(default_factory=list)
    _started: bool = False

    @classmethod
    def from_slurm(cls, config: RayClusterConfig) -> RayCluster:
        """Create a RayCluster from SLURM environment variables.

        This should be called inside a SLURM job where SLURM_JOB_NODELIST
        and related variables are set.
        """
        node_list = cls._get_slurm_nodes()
        head_ip = cls._get_node_ip(node_list[0], config.srun_export_env)
        return cls(config=config, head_ip=head_ip, node_list=node_list)

    @classmethod
    def from_hpc(cls, hpc: "HPC", num_nodes: int) -> RayCluster:
        """Create a RayCluster from an HPC configuration.

        Convenience method that extracts Ray-relevant settings from HPC.
        """
        ray_config = RayClusterConfig(
            num_nodes=num_nodes,
            gpus_per_node=hpc.gpus_per_node,
            cpus_per_node=hpc.cpus_per_node,
            srun_export_env=hpc.get_srun_export_env(),
            ray_env_vars=hpc.get_ray_env_vars(),
        )
        return cls.from_slurm(ray_config)

    @staticmethod
    def _get_slurm_nodes() -> List[str]:
        """Get list of node hostnames from SLURM environment."""
        nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
        if not nodelist:
            raise RuntimeError(
                "SLURM_JOB_NODELIST not set. Are you running inside a SLURM job?"
            )
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            check=True,
        )
        nodes = result.stdout.strip().split("\n")
        if not nodes or nodes == [""]:
            raise RuntimeError(f"No nodes found in SLURM_JOB_NODELIST: {nodelist}")
        return nodes

    @staticmethod
    def _get_node_ip(node: str, srun_export: str) -> str:
        """Get IP address for a node using srun."""
        result = subprocess.run(
            [
                "srun",
                f"--export={srun_export}",
                "--nodes=1",
                "--ntasks=1",
                "--overlap",
                "-w",
                node,
                "hostname",
                "--ip-address",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # hostname --ip-address can return multiple IPs; take the first
        return result.stdout.strip().split()[0]

    @property
    def address(self) -> str:
        """Ray cluster address in the format host:port."""
        return f"{self.head_ip}:{self.config.ray_port}"

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs across all nodes."""
        return len(self.node_list) * self.config.gpus_per_node

    @property
    def total_nodes(self) -> int:
        """Number of nodes in the cluster."""
        return len(self.node_list)

    def _cleanup_existing_ray(self) -> None:
        """Stop any existing Ray instances on allocated nodes.

        This ensures a clean slate before starting a new cluster,
        preventing conflicts with lingering processes from previous jobs.
        """
        print("Cleaning up existing Ray instances...")
        for node in self.node_list:
            try:
                subprocess.run(
                    [
                        "srun",
                        f"--export={self.config.srun_export_env}",
                        "--nodes=1",
                        "--ntasks=1",
                        "--overlap",
                        "-w",
                        node,
                        "ray",
                        "stop",
                        "--force",
                    ],
                    capture_output=True,
                    timeout=30,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                pass  # Ignore errors - node may not have Ray running
        # Brief pause to let processes terminate
        time.sleep(2)

    def start(self) -> None:
        """Start the Ray cluster.

        Starts the head node first, then worker nodes with a small delay
        between each to avoid overwhelming the head.
        """
        if self._started:
            print(f"Ray cluster already started at {self.address}")
            return

        # Clean up any lingering Ray instances from previous jobs
        self._cleanup_existing_ray()

        print(f"=== Starting Ray Cluster ===")
        print(f"  Nodes: {len(self.node_list)}")
        print(f"  GPUs per node: {self.config.gpus_per_node}")
        print(f"  CPUs per node: {self.config.cpus_per_node}")
        print(f"  Head node: {self.node_list[0]} ({self.head_ip})")
        print(f"  Ray port: {self.config.ray_port}")
        print(f"============================")

        # Start head node
        self._start_node(self.node_list[0], is_head=True)
        print(f"  Started Ray head on {self.node_list[0]}")

        # Start worker nodes with delay
        for i, node in enumerate(self.node_list[1:], start=1):
            self._start_node(node, is_head=False)
            print(f"  Started Ray worker {i} on {node}")
            time.sleep(3)  # Small delay between workers

        # Wait for cluster to be ready
        self._wait_for_cluster()
        self._started = True

        print(f"=== Ray Cluster Ready ===")
        print(f"  Address: {self.address}")
        print(f"  Total GPUs: {self.total_gpus}")
        print(f"=========================")

    def stop(self) -> None:
        """Stop the Ray cluster.

        Sends stop commands to all nodes and waits for processes to exit.
        """
        if not self._started and not self._ray_procs:
            return

        print("Stopping Ray cluster...")

        # Stop Ray on all nodes
        for node in self.node_list:
            try:
                subprocess.run(
                    [
                        "srun",
                        f"--export={self.config.srun_export_env}",
                        "--nodes=1",
                        "--ntasks=1",
                        "--overlap",
                        "-w",
                        node,
                        "ray",
                        "stop",
                        "--force",
                    ],
                    capture_output=True,
                    timeout=30,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                print(f"  Warning: Failed to stop Ray on {node}: {e}", file=sys.stderr)

        # Wait for background processes
        for proc in self._ray_procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        self._ray_procs.clear()
        self._ray_pids.clear()
        self._started = False
        print("Ray cluster stopped")

    def _start_node(self, node: str, is_head: bool) -> None:
        """Start Ray on a single node."""
        if is_head:
            cmd = [
                "ray",
                "start",
                "--head",
                f"--node-ip-address={self.head_ip}",
                f"--port={self.config.ray_port}",
                f"--num-gpus={self.config.gpus_per_node}",
                f"--num-cpus={self.config.cpus_per_node}",
                "--block",
            ]
        else:
            cmd = [
                "ray",
                "start",
                f"--address={self.address}",
                f"--num-gpus={self.config.gpus_per_node}",
                f"--num-cpus={self.config.cpus_per_node}",
                "--block",
            ]

        # Build the bash command with environment variables
        if self.config.ray_env_vars:
            bash_cmd = f"env {self.config.ray_env_vars} {' '.join(cmd)}"
        else:
            bash_cmd = " ".join(cmd)

        srun_cmd = [
            "srun",
            f"--export={self.config.srun_export_env}",
            "--nodes=1",
            "--ntasks=1",
            "--overlap",
            "-w",
            node,
            "bash",
            "-c",
            bash_cmd,
        ]

        proc = subprocess.Popen(
            srun_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._ray_procs.append(proc)
        self._ray_pids.append(proc.pid)

    def _wait_for_cluster(self) -> None:
        """Wait for the Ray cluster to be ready with expected resources."""
        script_path = Path(self.config.wait_for_cluster_script)

        if not script_path.exists():
            print(
                f"  Warning: {script_path} not found, using fallback wait",
                file=sys.stderr,
            )
            self._fallback_wait()
            return

        cmd = [
            sys.executable,
            str(script_path),
            "--address",
            self.address,
            "--expected-gpus",
            str(self.total_gpus),
            "--expected-nodes",
            str(len(self.node_list)),
            "--timeout",
            str(self.config.startup_timeout),
            "--poll-interval",
            str(self.config.poll_interval),
        ]

        print(f"  Waiting for cluster ({self.total_gpus} GPUs, {len(self.node_list)} nodes)...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Ray cluster failed to start within {self.config.startup_timeout}s"
            ) from e

    def _fallback_wait(self) -> None:
        """Fallback wait using ray.init() to check cluster status."""
        import ray

        start_time = time.time()
        while time.time() - start_time < self.config.startup_timeout:
            try:
                ray.init(address=self.address, ignore_reinit_error=True)
                resources = ray.cluster_resources()
                gpu_count = resources.get("GPU", 0)
                if gpu_count >= self.total_gpus:
                    print(f"  Cluster ready with {gpu_count} GPUs")
                    ray.shutdown()
                    return
                ray.shutdown()
            except Exception:
                pass
            time.sleep(self.config.poll_interval)

        raise RuntimeError(
            f"Ray cluster failed to reach {self.total_gpus} GPUs "
            f"within {self.config.startup_timeout}s"
        )

    def __enter__(self) -> RayCluster:
        """Context manager entry - start the cluster."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop the cluster."""
        self.stop()


def create_ray_cluster_from_slurm(
    gpus_per_node: int,
    cpus_per_node: int,
    ray_port: int = 6379,
    srun_export_env: str = "ALL",
    ray_env_vars: str = "",
) -> RayCluster:
    """Convenience function to create a Ray cluster from SLURM environment.

    This reads SLURM_JOB_NUM_NODES to determine the number of nodes.

    Args:
        gpus_per_node: Number of GPUs per node
        cpus_per_node: Number of CPUs per node
        ray_port: Port for Ray head node (default: 6379)
        srun_export_env: Environment export string for srun
        ray_env_vars: Space-separated KEY=value pairs for Ray workers

    Returns:
        A RayCluster configured from SLURM environment
    """
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))

    config = RayClusterConfig(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        cpus_per_node=cpus_per_node,
        ray_port=ray_port,
        srun_export_env=srun_export_env,
        ray_env_vars=ray_env_vars,
    )

    return RayCluster.from_slurm(config)
