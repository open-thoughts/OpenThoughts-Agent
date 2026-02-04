"""Beta9 Helm deployment to Kubernetes cluster."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import yaml

from scripts.beam.config import Beta9Config, S3StorageConfig

logger = logging.getLogger(__name__)

# Global to track cloned repo directory
_beta9_repo_dir: Optional[Path] = None

# Global to track log collector
_log_collector: Optional["LogCollector"] = None


class LogCollector:
    """Background log collector for Beta9 deployment debugging."""

    def __init__(self, namespace: str, output_dir: Path):
        self.namespace = namespace
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._log_file = self.output_dir / f"beta9_deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def start(self):
        """Start collecting logs in background threads."""
        logger.info(f"Starting log collection -> {self._log_file}")

        # Thread to collect pod status
        status_thread = threading.Thread(target=self._collect_pod_status, daemon=True)
        status_thread.start()
        self._threads.append(status_thread)

        # Thread to collect events
        events_thread = threading.Thread(target=self._collect_events, daemon=True)
        events_thread.start()
        self._threads.append(events_thread)

        # Thread to collect pod logs (continuously polls all containers)
        logs_thread = threading.Thread(target=self._collect_pod_logs, daemon=True)
        logs_thread.start()
        self._threads.append(logs_thread)

        # Thread to collect pod descriptions (for debugging crash reasons)
        describe_thread = threading.Thread(target=self._collect_pod_describe, daemon=True)
        describe_thread.start()
        self._threads.append(describe_thread)

    def stop(self):
        """Stop all collection threads."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5)
        logger.info(f"Log collection stopped. Logs saved to: {self._log_file}")

    def _write_log(self, section: str, content: str):
        """Write content to log file with timestamp."""
        with open(self._log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{datetime.now().isoformat()}] {section}\n")
            f.write(f"{'='*60}\n")
            f.write(content)
            f.write("\n")

    def _collect_pod_status(self):
        """Periodically collect pod status."""
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["kubectl", "get", "pods", "-n", self.namespace, "-o", "wide"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self._write_log("POD STATUS", result.stdout)
            except Exception as e:
                self._write_log("POD STATUS ERROR", str(e))

            self._stop_event.wait(30)  # Check every 30 seconds

    def _collect_events(self):
        """Periodically collect Kubernetes events."""
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["kubectl", "get", "events", "-n", self.namespace, "--sort-by=.lastTimestamp"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self._write_log("EVENTS", result.stdout)
            except Exception as e:
                self._write_log("EVENTS ERROR", str(e))

            self._stop_event.wait(30)

    def _collect_pod_logs(self):
        """Continuously collect logs from all pods."""
        last_log_hash: dict[str, str] = {}  # Track log content to avoid duplicates

        while not self._stop_event.is_set():
            try:
                # Get list of pods with their container info (including init containers)
                result = subprocess.run(
                    ["kubectl", "get", "pods", "-n", self.namespace, "-o",
                     "jsonpath={range .items[*]}{.metadata.name},{.status.phase},{.status.containerStatuses[*].name},{.status.containerStatuses[*].restartCount},{.status.initContainerStatuses[*].name}|{end}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0 and result.stdout:
                    for pod_info in result.stdout.split("|"):
                        if not pod_info.strip():
                            continue
                        parts = pod_info.split(",")
                        if len(parts) >= 2:
                            pod_name = parts[0]
                            pod_phase = parts[1]
                            containers = parts[2].split() if len(parts) > 2 else []
                            restart_counts = parts[3].split() if len(parts) > 3 else []
                            init_containers = parts[4].split() if len(parts) > 4 else []

                            # Collect logs for each init container (completed ones)
                            for init_container in init_containers:
                                self._collect_container_logs(
                                    pod_name,
                                    init_container,
                                    pod_phase,
                                    0,  # Init containers don't restart
                                    last_log_hash,
                                    is_init=True
                                )

                            # Collect logs for each container
                            for i, container in enumerate(containers):
                                self._collect_container_logs(
                                    pod_name,
                                    container,
                                    pod_phase,
                                    int(restart_counts[i]) if i < len(restart_counts) else 0,
                                    last_log_hash
                                )
            except Exception as e:
                self._write_log("POD LOG COLLECTION ERROR", str(e))

            self._stop_event.wait(15)  # Poll every 15 seconds

    def _collect_container_logs(
        self,
        pod_name: str,
        container: str,
        pod_phase: str,
        restart_count: int,
        last_log_hash: dict[str, str],
        is_init: bool = False
    ):
        """Collect logs from a specific container, including previous crash logs."""
        log_key = f"{pod_name}/{container}"

        try:
            # Get current logs
            result = subprocess.run(
                ["kubectl", "logs", "-n", self.namespace, pod_name, "-c", container, "--tail=200"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.stdout:
                # Hash to avoid duplicate entries
                log_hash = hash(result.stdout)
                if last_log_hash.get(log_key) != log_hash:
                    last_log_hash[log_key] = log_hash
                    container_type = "INIT" if is_init else "LOGS"
                    header = f"{container_type}: {pod_name} -c {container} (phase={pod_phase}, restarts={restart_count})"
                    self._write_log(header, result.stdout)

            # If container has restarted, also get previous logs
            if restart_count > 0:
                prev_key = f"{log_key}/previous"
                prev_result = subprocess.run(
                    ["kubectl", "logs", "-n", self.namespace, pod_name, "-c", container, "--previous", "--tail=200"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if prev_result.stdout:
                    prev_hash = hash(prev_result.stdout)
                    if last_log_hash.get(prev_key) != prev_hash:
                        last_log_hash[prev_key] = prev_hash
                        self._write_log(f"PREVIOUS LOGS: {pod_name} -c {container}", prev_result.stdout)

        except Exception as e:
            self._write_log(f"CONTAINER LOG ERROR: {pod_name}/{container}", str(e))

    def _collect_pod_describe(self):
        """Periodically collect pod descriptions for debugging."""
        described_pods: set[str] = set()

        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["kubectl", "get", "pods", "-n", self.namespace, "-o", "jsonpath={.items[*].metadata.name}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    for pod_name in result.stdout.split():
                        if pod_name and pod_name not in described_pods:
                            described_pods.add(pod_name)
                            describe_result = subprocess.run(
                                ["kubectl", "describe", "pod", "-n", self.namespace, pod_name],
                                capture_output=True,
                                text=True,
                                timeout=30,
                            )
                            if describe_result.stdout:
                                self._write_log(f"DESCRIBE: {pod_name}", describe_result.stdout)
            except Exception:
                pass

            self._stop_event.wait(60)  # Describe once per minute for new pods


def start_log_collection(namespace: str, output_dir: Optional[Path] = None) -> LogCollector:
    """Start background log collection.

    Args:
        namespace: Kubernetes namespace to monitor.
        output_dir: Directory to save logs. Defaults to ./beta9_logs/

    Returns:
        LogCollector instance.
    """
    global _log_collector

    if output_dir is None:
        output_dir = Path.cwd() / "beta9_logs"

    _log_collector = LogCollector(namespace, output_dir)
    _log_collector.start()
    return _log_collector


def stop_log_collection():
    """Stop background log collection."""
    global _log_collector
    if _log_collector:
        _log_collector.stop()
        _log_collector = None


def check_helm_installed() -> bool:
    """Check if Helm CLI is installed."""
    if not shutil.which("helm"):
        logger.error("helm not found. Install from: https://helm.sh/docs/intro/install/")
        return False
    return True


def add_helm_repo(dry_run: bool = False, config: Optional[Beta9Config] = None) -> bool:
    """Prepare the Helm chart for deployment.

    Uses local chart only. Remote chart usage is disallowed to avoid config drift.

    Args:
        dry_run: If True, print command without executing.
        config: Beta9 configuration (optional, uses defaults if not provided).

    Returns:
        True if chart is ready.
    """
    global _beta9_repo_dir

    if config is None:
        config = Beta9Config()

    if not config.use_local_chart:
        logger.error("Remote Helm charts are disabled. Set use_local_chart=True.")
        return False

    # Find the project root (where scripts/beam/helm-chart is)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # scripts/beam -> scripts -> project root
    local_chart_path = project_root / config.helm_chart_local_path

    if dry_run:
        logger.info(f"[DRY RUN] Would use local chart: {local_chart_path}")
        return True

    if not local_chart_path.exists():
        logger.error(f"Local Helm chart not found at: {local_chart_path}")
        return False

    # Set the repo dir to None to indicate we're using local chart
    _beta9_repo_dir = None
    logger.info(f"Using local Helm chart: {local_chart_path}")
    return True


def get_helm_chart_path(config: Beta9Config) -> Optional[Path]:
    """Get the path to the Helm chart (local or cloned).

    Args:
        config: Beta9 configuration.

    Returns:
        Path to the chart directory, or None if not available.
    """
    global _beta9_repo_dir

    # If using local chart
    if config.use_local_chart:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        local_chart_path = project_root / config.helm_chart_local_path
        if local_chart_path.exists():
            return local_chart_path
        return None

    # If using cloned repo
    if _beta9_repo_dir is None:
        return None

    return _beta9_repo_dir / config.helm_chart_git_path


def cleanup_helm_repo():
    """Clean up the cloned Beta9 repository."""
    global _beta9_repo_dir

    if _beta9_repo_dir is not None and _beta9_repo_dir.exists():
        shutil.rmtree(_beta9_repo_dir)
        logger.debug(f"Cleaned up temp repo: {_beta9_repo_dir}")
        _beta9_repo_dir = None


def create_namespace(namespace: str, dry_run: bool = False) -> bool:
    """Create Kubernetes namespace if it doesn't exist.

    Args:
        namespace: Namespace name.
        dry_run: If True, print command without executing.

    Returns:
        True if namespace exists or created successfully.
    """
    # Check if exists
    check_result = subprocess.run(
        ["kubectl", "get", "namespace", namespace],
        capture_output=True,
        text=True,
    )

    if check_result.returncode == 0:
        logger.debug(f"Namespace '{namespace}' already exists")
        return True

    cmd = ["kubectl", "create", "namespace", namespace]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to create namespace: {result.stderr}")
        return False

    logger.info(f"Created namespace '{namespace}'")
    return True


def build_helm_dependencies(config: Beta9Config, dry_run: bool = False) -> bool:
    """Build Helm chart dependencies.

    Args:
        config: Beta9 configuration.
        dry_run: If True, print command without executing.

    Returns:
        True if dependencies built successfully.
    """
    chart_path = get_helm_chart_path(config)
    if chart_path is None:
        logger.error("Helm chart not available. Run add_helm_repo() first.")
        return False

    chart_path_str = str(chart_path)

    cmd = ["helm", "dependency", "update", chart_path_str]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info("Updating Helm chart dependencies...")
    logger.info("  This may take a minute...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to build dependencies: {result.stderr}")
        return False

    logger.info("Helm dependencies updated successfully")
    return True


def _build_config_json(s3_config: Optional[S3StorageConfig]) -> str:
    """Build CONFIG_JSON for Beta9 gateway with S3 credentials.

    Args:
        s3_config: S3 storage config. If None, uses LocalStack defaults.

    Returns:
        JSON string for CONFIG_JSON environment variable.
    """
    # Base config with correct service names
    config = {
        "database": {
            "postgres": {
                "host": "postgresql",
                "port": 5432,
                "name": "main",
                "username": "root",
                "password": "password",
            },
            "redis": {
                "mode": "single",
                "addrs": ["redis-master:6379"],
            },
        },
        "storage": {
            "mode": "juicefs",
            "fsName": "beta9-fs",
            "fsPath": "/data",
            "objectPath": "/data/objects",
            "juicefs": {
                "redisURI": "redis://juicefs-redis-master:6379/0",
                "blockSize": 4096,
                "cacheSize": 1024,
                "prefetch": 1,
                "bufferSize": 300,
            },
            "workspaceStorage": {
                "defaultStorageMode": "s3",
                "defaultRegion": "us-east-1",
            },
        },
    }

    if s3_config:
        # Use external S3-compatible storage
        # NOTE: awsS3Bucket must be endpoint + bucket, NOT with extra path suffix!
        # JuiceFS manages its own directory structure using fsName (beta9-fs).
        # Format: https://endpoint:port/bucket (no extra path segments)
        juicefs_bucket_url = f"{s3_config.endpoint_url}/{s3_config.bucket_name}"

        logger.info(f"[CONFIG] Using EXTERNAL S3 storage (not LocalStack)")
        logger.info(f"[CONFIG]   JuiceFS bucket URL: {juicefs_bucket_url}")
        logger.info(f"[CONFIG]   Workspace endpoint: {s3_config.endpoint_url}")

        config["storage"]["juicefs"].update({
            "awsS3Bucket": juicefs_bucket_url,
            "awsAccessKey": s3_config.access_key,
            "awsSecretKey": s3_config.secret_key,
            "storageType": "minio",  # Use minio for S3-compatible storage (not AWS S3)
        })

        # Workspace storage (container images) - uses separate prefix
        config["storage"]["workspaceStorage"].update({
            "defaultBucketPrefix": "beta9-workspace",
            "defaultAccessKey": s3_config.access_key,
            "defaultSecretKey": s3_config.secret_key,
            "defaultEndpointUrl": s3_config.endpoint_url,
        })
    else:
        # Fall back to LocalStack (for testing without external S3)
        # NOTE: awsS3Bucket must be a FULL URL (endpoint + bucket)
        logger.warning("[CONFIG] FALLBACK: No S3 config provided, using LocalStack defaults!")
        logger.warning("[CONFIG]   JuiceFS bucket URL: http://localstack:4566/juicefs")
        logger.warning("[CONFIG]   This will FAIL if LocalStack is not deployed!")

        config["storage"]["juicefs"].update({
            "awsS3Bucket": "http://localstack:4566/juicefs",
            "awsAccessKey": "test",
            "awsSecretKey": "test",
        })
        config["storage"]["workspaceStorage"].update({
            "defaultBucketPrefix": "beta9-images",
            "defaultAccessKey": "test",
            "defaultSecretKey": "test",
            "defaultEndpointUrl": "http://localstack:4566",
        })

    return json.dumps(config)


def deploy_beta9(
    config: Beta9Config,
    dry_run: bool = False,
    s3_config: Optional[S3StorageConfig] = None,
) -> bool:
    """Deploy Beta9 to the cluster using Helm.

    Args:
        config: Beta9 configuration.
        dry_run: If True, print command without executing.
        s3_config: Optional S3 storage config for external MinIO. If provided,
                   Beta9 will use this for shared storage instead of LocalStack.

    Returns:
        True if deployment successful.
    """
    # Create namespace first
    if not create_namespace(config.namespace, dry_run):
        return False

    # Get the local chart path
    chart_path = get_helm_chart_path(config)
    if chart_path is None:
        logger.error("Helm chart not available. Run add_helm_repo() first.")
        return False

    chart_path_str = str(chart_path)

    # Build dependencies first
    if not build_helm_dependencies(config, dry_run):
        return False

    # Build Helm values for CPU-only deployment
    # NOTE: We inject S3 config directly into the Helm values (config section) instead of
    # using CONFIG_JSON env var. This ensures the mounted config file has the S3 credentials.
    # Beta9's koanf config merger wasn't properly merging CONFIG_JSON with the mounted config.
    values = {
        "gateway": {
            "replicas": 1,
            "service": {
                "type": "ClusterIP",
                "httpPort": config.gateway_http_port,
                "grpcPort": config.gateway_grpc_port,
            },
        },
        "worker": {
            "enabled": True,
            "replicas": config.worker_replicas,
            "resources": {
                "requests": {
                    "cpu": config.worker_cpu_request,
                    "memory": config.worker_memory_request,
                },
            },
        },
        # Disable GPU-specific components
        "gpu": {
            "enabled": False,
        },
    }

    # Inject S3 config directly into the Helm values (goes into mounted config file)
    if s3_config:
        # Format: https://endpoint:port/bucket (no extra path - JuiceFS uses fsName for internal paths)
        juicefs_bucket_url = f"{s3_config.endpoint_url}/{s3_config.bucket_name}"

        logger.info(f"[CONFIG] Using EXTERNAL S3 storage (not LocalStack)")
        logger.info(f"[CONFIG]   JuiceFS bucket URL: {juicefs_bucket_url}")
        logger.info(f"[CONFIG]   JuiceFS fsName: {config.juicefs_fs_name}")
        logger.info(f"[CONFIG]   Workspace endpoint: {s3_config.endpoint_url}")

        # Add S3 config to Helm values - this updates the mounted config file
        values["config"] = {
            "debugMode": True,  # Enable debug logging to see what config is loaded
            "storage": {
                "fsName": config.juicefs_fs_name,  # Configurable JuiceFS prefix
                "juicefs": {
                    "awsS3Bucket": juicefs_bucket_url,
                    "awsAccessKey": s3_config.access_key,
                    "awsSecretKey": s3_config.secret_key,
                    "storageType": "minio",  # Use minio for S3-compatible storage
                },
                "workspaceStorage": {
                    "defaultAccessKey": s3_config.access_key,
                    "defaultSecretKey": s3_config.secret_key,
                    "defaultEndpointUrl": s3_config.endpoint_url,
                },
            },
        }
    else:
        logger.warning("[CONFIG] FALLBACK: No S3 config provided, using LocalStack defaults!")
        logger.warning("[CONFIG]   JuiceFS bucket URL: http://localstack:4566/juicefs")
        logger.warning("[CONFIG]   JuiceFS fsName: {config.juicefs_fs_name}")
        logger.warning("[CONFIG]   This will FAIL if LocalStack is not deployed!")

        values["config"] = {
            "debugMode": True,  # Enable debug logging to see what config is loaded
            "storage": {
                "fsName": config.juicefs_fs_name,  # Configurable JuiceFS prefix
                "juicefs": {
                    "awsS3Bucket": "http://localstack:4566/juicefs",
                    "awsAccessKey": "test",
                    "awsSecretKey": "test",
                },
                "workspaceStorage": {
                    "defaultAccessKey": "test",
                    "defaultSecretKey": "test",
                    "defaultEndpointUrl": "http://localstack:4566",
                },
            },
        }

    # Write values to a temp file and use --values instead of --set-json
    # This avoids parsing issues with --set-json and ensures proper deep merging
    # with the chart's values.yaml
    if dry_run:
        logger.info(f"[DRY RUN] Would deploy with values:")
        logger.info(yaml.dump(values, default_flow_style=False))
        return True

    logger.info(f"Deploying Beta9 to namespace '{config.namespace}'...")
    logger.info(f"  Using local chart from: {chart_path_str}")
    logger.info("  Helm install will return quickly; waiting handled separately...")

    # Start background log collection for debugging (kept running for wait_for_gateway_ready)
    start_log_collection(config.namespace)

    # Write values to temp file (Helm --values properly deep-merges with values.yaml)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(values, f, default_flow_style=False)
        values_file = f.name

    try:
        actual_cmd = [
            "helm", "upgrade", "--install",
            config.helm_release_name,
            chart_path_str,
            "--namespace", config.namespace,
            "--values", values_file,
            # No --wait: let wait_for_gateway_ready() handle waiting with crash detection
        ]

        logger.debug(f"Helm command: {' '.join(actual_cmd)}")
        logger.debug(f"Values file: {values_file}")

        result = subprocess.run(actual_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Failed to deploy Beta9: {result.stderr}")
            logger.error(f"stdout: {result.stdout}")
            stop_log_collection()
            return False

        logger.info("Helm install submitted successfully (pods starting...)")
        return True
    finally:
        # Clean up temp file
        try:
            os.unlink(values_file)
        except OSError:
            pass


def uninstall_beta9(config: Beta9Config, dry_run: bool = False) -> bool:
    """Uninstall Beta9 from the cluster.

    Args:
        config: Beta9 configuration.
        dry_run: If True, print command without executing.

    Returns:
        True if uninstall successful.
    """
    cmd = [
        "helm", "uninstall",
        config.helm_release_name,
        "--namespace", config.namespace,
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Uninstalling Beta9 from namespace '{config.namespace}'...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 and "not found" not in result.stderr.lower():
        logger.error(f"Failed to uninstall Beta9: {result.stderr}")
        return False

    logger.info("Beta9 uninstalled successfully")
    return True


def delete_namespace_pvcs(namespace: str, dry_run: bool = False) -> bool:
    """Delete all PersistentVolumeClaims in the namespace.

    Helm uninstall does NOT delete PVCs by default (to protect data).
    This function explicitly deletes all PVCs in the namespace to clean up
    the associated GCP persistent disks.

    Args:
        namespace: Kubernetes namespace.
        dry_run: If True, print command without executing.

    Returns:
        True if PVCs deleted successfully (or none existed).
    """
    # First, list PVCs to show what will be deleted
    list_cmd = [
        "kubectl", "get", "pvc",
        "--namespace", namespace,
        "-o", "jsonpath={.items[*].metadata.name}",
    ]

    result = subprocess.run(list_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Namespace might not exist
        logger.debug(f"Could not list PVCs: {result.stderr}")
        return True

    pvcs = result.stdout.strip().split()
    if not pvcs or pvcs == ['']:
        logger.info("No PVCs to delete in namespace")
        return True

    logger.info(f"Found {len(pvcs)} PVCs to delete: {', '.join(pvcs)}")

    if dry_run:
        logger.info(f"[DRY RUN] Would delete {len(pvcs)} PVCs")
        return True

    # Delete all PVCs in the namespace (with timeout to avoid hanging)
    delete_cmd = [
        "kubectl", "delete", "pvc", "--all",
        "--namespace", namespace,
        "--timeout=60s",  # Don't wait forever
    ]

    logger.info(f"Deleting all PVCs in namespace '{namespace}'...")
    result = subprocess.run(delete_cmd, capture_output=True, text=True, timeout=90)

    if result.returncode != 0:
        # PVCs may be stuck due to finalizers - force delete them
        logger.warning(f"PVC deletion timed out, removing finalizers...")
        for pvc in pvcs:
            patch_cmd = [
                "kubectl", "patch", "pvc", pvc,
                "--namespace", namespace,
                "-p", '{"metadata":{"finalizers":null}}',
            ]
            subprocess.run(patch_cmd, capture_output=True, text=True, timeout=30)
        logger.info(f"Forced deletion of {len(pvcs)} PVCs")
        return True

    logger.info(f"Deleted {len(pvcs)} PVCs successfully")
    return True


def get_gateway_cluster_ip(config: Beta9Config) -> Optional[str]:
    """Get the ClusterIP of the Beta9 gateway service.

    Args:
        config: Beta9 configuration.

    Returns:
        ClusterIP address or None if not found.
    """
    result = subprocess.run(
        [
            "kubectl", "get", "service",
            config.gateway_service_name,
            "--namespace", config.namespace,
            "-o", "jsonpath={.spec.clusterIP}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not result.stdout.strip():
        return None

    return result.stdout.strip()


def wait_for_gateway_ready(
    config: Beta9Config,
    timeout: int = 300,
    check_endpoint: bool = True,
) -> bool:
    """Wait for Beta9 gateway to be ready.

    Args:
        config: Beta9 configuration.
        timeout: Maximum seconds to wait.
        check_endpoint: If True, also check HTTP health endpoint.

    Returns:
        True if gateway is ready within timeout.
    """
    global _log_collector

    logger.info("Waiting for Beta9 gateway to be ready...")
    start_time = time.time()

    # Start log collection if not already running (to capture crash logs)
    if _log_collector is None:
        start_log_collection(config.namespace)

    crash_count = 0
    max_crash_retries = 5  # Give up after 5 consecutive CrashLoopBackOff detections

    while time.time() - start_time < timeout:
        # Check pod status with more detail
        # Label selector: app.kubernetes.io/name=beta9,app.kubernetes.io/component=gateway
        result = subprocess.run(
            [
                "kubectl", "get", "pods",
                "--namespace", config.namespace,
                "-l", f"app.kubernetes.io/name={config.helm_release_name},app.kubernetes.io/component=gateway",
                "-o", "jsonpath={range .items[*]}{.metadata.name}:{.status.phase}:{.status.containerStatuses[*].state.waiting.reason}:{.status.containerStatuses[*].restartCount}{\"\\n\"}{end}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.split(":")
                if len(parts) >= 4:
                    pod_name, phase, waiting_reason, restart_count = parts[0], parts[1], parts[2], parts[3]

                    # Check for CrashLoopBackOff
                    if waiting_reason == "CrashLoopBackOff":
                        crash_count += 1
                        logger.warning(f"Gateway pod in CrashLoopBackOff (restarts: {restart_count}, detection: {crash_count}/{max_crash_retries})")

                        if crash_count >= max_crash_retries:
                            logger.error("Gateway pod is stuck in CrashLoopBackOff")
                            logger.error("Check logs at: beta9_logs/ for debugging info")
                            return False

                    elif phase == "Running" and not waiting_reason:
                        # Check if container is actually ready
                        ready_result = subprocess.run(
                            [
                                "kubectl", "get", "pods",
                                "--namespace", config.namespace,
                                pod_name,
                                "-o", "jsonpath={.status.containerStatuses[*].ready}",
                            ],
                            capture_output=True,
                            text=True,
                        )

                        if "true" in ready_result.stdout.lower():
                            logger.info("Gateway pod is running and ready")
                            return True
                        else:
                            logger.debug(f"Gateway pod running but not ready yet")
                            crash_count = 0  # Reset crash counter if we see progress

        logger.debug(f"Gateway status: {result.stdout.strip()}")
        time.sleep(10)

    logger.error(f"Gateway did not become ready within {timeout} seconds")
    return False


def get_deployment_status(config: Beta9Config) -> dict:
    """Get status of Beta9 deployment.

    Returns:
        Dictionary with deployment status information.
    """
    status = {
        "installed": False,
        "gateway_ready": False,
        "worker_ready": False,
        "pods": [],
    }

    # Check if Helm release exists
    result = subprocess.run(
        [
            "helm", "status",
            config.helm_release_name,
            "--namespace", config.namespace,
            "-o", "json",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        status["installed"] = True

    # Get pod status
    pod_result = subprocess.run(
        [
            "kubectl", "get", "pods",
            "--namespace", config.namespace,
            "-o", "json",
        ],
        capture_output=True,
        text=True,
    )

    if pod_result.returncode == 0:
        try:
            pods_data = json.loads(pod_result.stdout)
            for pod in pods_data.get("items", []):
                pod_info = {
                    "name": pod["metadata"]["name"],
                    "phase": pod["status"]["phase"],
                    "ready": all(
                        c.get("ready", False)
                        for c in pod["status"].get("containerStatuses", [])
                    ),
                }
                status["pods"].append(pod_info)

                if "gateway" in pod_info["name"] and pod_info["ready"]:
                    status["gateway_ready"] = True
                if "worker" in pod_info["name"] and pod_info["ready"]:
                    status["worker_ready"] = True
        except json.JSONDecodeError:
            pass

    return status
