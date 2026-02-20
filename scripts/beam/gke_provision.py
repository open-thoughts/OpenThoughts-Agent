"""GKE cluster provisioning using gcloud CLI."""

import logging
import shutil
import subprocess
import time
from typing import Optional

from scripts.beam.config import GKEConfig, FilestoreConfig

logger = logging.getLogger(__name__)


def check_gcloud_installed() -> bool:
    """Check if gcloud CLI is installed and authenticated."""
    if not shutil.which("gcloud"):
        logger.error("gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install")
        return False

    # Check authentication
    result = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        logger.error("gcloud not authenticated. Run: gcloud auth login")
        return False

    return True


def check_kubectl_installed() -> bool:
    """Check if kubectl is installed."""
    if not shutil.which("kubectl"):
        logger.error("kubectl not found. Install from: https://kubernetes.io/docs/tasks/tools/")
        return False
    return True


def check_git_installed() -> bool:
    """Check if git is installed."""
    if not shutil.which("git"):
        logger.error("git not found. Install from: https://git-scm.com/downloads")
        return False
    return True


def cluster_exists(config: GKEConfig) -> bool:
    """Check if the GKE cluster already exists."""
    result = subprocess.run(
        [
            "gcloud", "container", "clusters", "describe",
            config.cluster_name,
            "--project", config.project_id,
            "--zone", config.zone,  # Zonal cluster
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and config.cluster_name in result.stdout


def create_cluster(config: GKEConfig, dry_run: bool = False) -> bool:
    """Create a GKE cluster with CPU-only nodes.

    Args:
        config: GKE configuration.
        dry_run: If True, print command without executing.

    Returns:
        True if cluster created successfully (or dry_run).
    """
    # Use --zone for a zonal cluster (exact num_nodes) instead of --region
    # Regional clusters create num_nodes PER ZONE (3 zones = 3x nodes)
    #
    # IMPORTANT: Use UBUNTU_CONTAINERD image type instead of default COS (Container-Optimized OS)
    # because Beta9 workers require hostPath volume mounts at /images and /data/checkpoints.
    # COS has a read-only root filesystem which causes "mkdir /images: read-only file system" errors.
    # Ubuntu nodes have a writable root filesystem that allows these hostPath mounts.
    cmd = [
        "gcloud", "container", "clusters", "create", config.cluster_name,
        "--project", config.project_id,
        "--zone", config.zone,  # Zonal cluster: exactly num_nodes total
        "--num-nodes", str(config.num_nodes),
        "--machine-type", config.machine_type,
        "--image-type", "UBUNTU_CONTAINERD",  # Required: writable root FS for Beta9 hostPath mounts
        "--disk-size", f"{config.disk_size_gb}GB",
        "--network", config.network,
        "--no-enable-autoupgrade",
        "--no-enable-autorepair",
    ]

    if config.enable_ip_alias:
        cmd.append("--enable-ip-alias")

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Creating GKE cluster '{config.cluster_name}' with {config.num_nodes} nodes...")
    logger.info(f"  Machine type: {config.machine_type}")
    logger.info(f"  Image type: UBUNTU_CONTAINERD (required for Beta9 hostPath mounts)")
    logger.info(f"  Zone: {config.zone} (zonal cluster)")
    logger.info("  This may take 5-10 minutes...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to create cluster: {result.stderr}")
        return False

    logger.info(f"Cluster '{config.cluster_name}' created successfully")
    return True


def delete_cluster(config: GKEConfig, dry_run: bool = False) -> bool:
    """Delete the GKE cluster.

    Args:
        config: GKE configuration.
        dry_run: If True, print command without executing.

    Returns:
        True if cluster deleted successfully (or dry_run).
    """
    cmd = [
        "gcloud", "container", "clusters", "delete", config.cluster_name,
        "--project", config.project_id,
        "--zone", config.zone,  # Zonal cluster
        "--quiet",  # Skip confirmation prompt
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Deleting GKE cluster '{config.cluster_name}'...")
    logger.info("  This may take a few minutes...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to delete cluster: {result.stderr}")
        return False

    logger.info(f"Cluster '{config.cluster_name}' deleted successfully")
    return True


def get_credentials(config: GKEConfig, dry_run: bool = False) -> bool:
    """Get kubectl credentials for the GKE cluster.

    Args:
        config: GKE configuration.
        dry_run: If True, print command without executing.

    Returns:
        True if credentials obtained successfully (or dry_run).
    """
    cmd = [
        "gcloud", "container", "clusters", "get-credentials", config.cluster_name,
        "--project", config.project_id,
        "--zone", config.zone,  # Zonal cluster
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Getting kubectl credentials for '{config.cluster_name}'...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to get credentials: {result.stderr}")
        return False

    logger.info("kubectl credentials configured successfully")
    return True


def grant_cluster_admin(dry_run: bool = False) -> bool:
    """Grant cluster-admin role to the current gcloud user.

    This is required for deploying Helm charts that create ClusterRoles
    and ClusterRoleBindings (like Beta9).

    Args:
        dry_run: If True, print command without executing.

    Returns:
        True if successful or binding already exists.
    """
    # Get current gcloud user
    user_result = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True,
        text=True,
    )

    if user_result.returncode != 0 or not user_result.stdout.strip():
        logger.error("Could not determine current gcloud user")
        return False

    user_email = user_result.stdout.strip()

    cmd = [
        "kubectl", "create", "clusterrolebinding", "cluster-admin-binding",
        "--clusterrole=cluster-admin",
        f"--user={user_email}",
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Granting cluster-admin to {user_email}...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # "already exists" is fine
    if result.returncode != 0:
        if "already exists" in result.stderr:
            logger.info("cluster-admin binding already exists")
            return True
        logger.error(f"Failed to grant cluster-admin: {result.stderr}")
        return False

    logger.info("cluster-admin granted successfully")
    return True


def get_cluster_status(config: GKEConfig) -> Optional[str]:
    """Get the current status of the cluster.

    Returns:
        Cluster status string (e.g., "RUNNING", "PROVISIONING") or None if not found.
    """
    result = subprocess.run(
        [
            "gcloud", "container", "clusters", "describe",
            config.cluster_name,
            "--project", config.project_id,
            "--zone", config.zone,  # Zonal cluster
            "--format=value(status)",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def wait_for_cluster_ready(config: GKEConfig, timeout: int = 600) -> bool:
    """Wait for cluster to reach RUNNING status.

    Args:
        config: GKE configuration.
        timeout: Maximum seconds to wait.

    Returns:
        True if cluster is running within timeout.
    """
    logger.info("Waiting for cluster to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        status = get_cluster_status(config)
        if status == "RUNNING":
            logger.info("Cluster is ready")
            return True

        logger.debug(f"Cluster status: {status}")
        time.sleep(10)

    logger.error(f"Cluster did not become ready within {timeout} seconds")
    return False


def get_node_count(config: GKEConfig) -> int:
    """Get current number of nodes in the cluster."""
    result = subprocess.run(
        ["kubectl", "get", "nodes", "-o", "name"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return 0

    return len(result.stdout.strip().split("\n"))


# =============================================================================
# Filestore (NFS) Functions for ReadWriteMany Storage
# =============================================================================


def filestore_exists(config: FilestoreConfig, gke_config: GKEConfig) -> bool:
    """Check if the Filestore instance already exists."""
    result = subprocess.run(
        [
            "gcloud", "filestore", "instances", "describe",
            config.instance_name,
            "--project", gke_config.project_id,
            "--zone", gke_config.zone,
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def create_filestore(
    config: FilestoreConfig, gke_config: GKEConfig, dry_run: bool = False,
    timeout: int = 600
) -> bool:
    """Create a Filestore instance for shared NFS storage.

    Filestore provides ReadWriteMany (RWX) storage that can be mounted by
    multiple pods simultaneously, which is required for Beta9's shared
    container image storage.

    Args:
        config: Filestore configuration.
        gke_config: GKE configuration (for project/zone).
        dry_run: If True, print command without executing.
        timeout: Maximum seconds to wait for creation (default 10 minutes).

    Returns:
        True if Filestore created successfully (or dry_run).
    """
    cmd = [
        "gcloud", "filestore", "instances", "create", config.instance_name,
        "--project", gke_config.project_id,
        "--zone", gke_config.zone,
        "--tier", config.tier,
        "--file-share", f"name={config.file_share_name},capacity={config.capacity_gb}GB",
        "--network", f"name={config.network}",
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Creating Filestore instance '{config.instance_name}'...")
    logger.info(f"  Tier: {config.tier}")
    logger.info(f"  Capacity: {config.capacity_gb}GB")
    logger.info(f"  Cost: ~${config.capacity_gb * 0.20 / 730:.2f}/hour (Basic HDD)")
    logger.info(f"  This may take 2-5 minutes (timeout: {timeout}s)...")

    # Use Popen to show progress while waiting
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Poll for completion with progress logging
    last_log_time = start_time
    while True:
        try:
            # Check if process finished (non-blocking with short timeout)
            stdout, stderr = process.communicate(timeout=10)
            # Process finished
            elapsed = int(time.time() - start_time)
            if process.returncode != 0:
                logger.error(f"Failed to create Filestore after {elapsed}s: {stderr}")
                return False
            logger.info(f"Filestore '{config.instance_name}' created successfully ({elapsed}s)")
            return True
        except subprocess.TimeoutExpired:
            # Still running - log progress
            elapsed = int(time.time() - start_time)
            if elapsed >= timeout:
                logger.error(f"Filestore creation timed out after {timeout}s")
                process.kill()
                return False
            # Log progress every 30 seconds
            if time.time() - last_log_time >= 30:
                logger.info(f"  Still creating Filestore... ({elapsed}s elapsed)")
                last_log_time = time.time()


def delete_filestore(
    config: FilestoreConfig, gke_config: GKEConfig, dry_run: bool = False,
    timeout: int = 300
) -> bool:
    """Delete the Filestore instance.

    Args:
        config: Filestore configuration.
        gke_config: GKE configuration (for project/zone).
        dry_run: If True, print command without executing.
        timeout: Maximum seconds to wait for deletion (default 5 minutes).

    Returns:
        True if Filestore deleted successfully (or dry_run).
    """
    cmd = [
        "gcloud", "filestore", "instances", "delete", config.instance_name,
        "--project", gke_config.project_id,
        "--zone", gke_config.zone,
        "--quiet",  # Skip confirmation prompt
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Deleting Filestore instance '{config.instance_name}'...")

    # Use Popen to show progress while waiting
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Poll for completion with progress logging
    last_log_time = start_time
    while True:
        try:
            stdout, stderr = process.communicate(timeout=10)
            elapsed = int(time.time() - start_time)
            if process.returncode != 0:
                # Don't fail if it doesn't exist
                if "not found" in stderr.lower() or "does not exist" in stderr.lower():
                    logger.info(f"Filestore '{config.instance_name}' does not exist (already deleted)")
                    return True
                logger.error(f"Failed to delete Filestore after {elapsed}s: {stderr}")
                return False
            logger.info(f"Filestore '{config.instance_name}' deleted successfully ({elapsed}s)")
            return True
        except subprocess.TimeoutExpired:
            elapsed = int(time.time() - start_time)
            if elapsed >= timeout:
                logger.error(f"Filestore deletion timed out after {timeout}s")
                process.kill()
                return False
            if time.time() - last_log_time >= 30:
                logger.info(f"  Still deleting Filestore... ({elapsed}s elapsed)")
                last_log_time = time.time()


def get_filestore_ip(config: FilestoreConfig, gke_config: GKEConfig) -> Optional[str]:
    """Get the IP address of the Filestore instance.

    Args:
        config: Filestore configuration.
        gke_config: GKE configuration (for project/zone).

    Returns:
        IP address string or None if not found.
    """
    result = subprocess.run(
        [
            "gcloud", "filestore", "instances", "describe",
            config.instance_name,
            "--project", gke_config.project_id,
            "--zone", gke_config.zone,
            "--format=value(networks[0].ipAddresses[0])",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Failed to get Filestore IP: {result.stderr}")
        return None

    ip = result.stdout.strip()
    if ip:
        config.ip_address = ip
        logger.info(f"Filestore IP: {ip}")
    return ip


def wait_for_filestore_ready(
    config: FilestoreConfig, gke_config: GKEConfig, timeout: int = 300
) -> bool:
    """Wait for Filestore instance to be ready.

    Args:
        config: Filestore configuration.
        gke_config: GKE configuration (for project/zone).
        timeout: Maximum seconds to wait.

    Returns:
        True if Filestore is ready within timeout.
    """
    logger.info("Waiting for Filestore to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        result = subprocess.run(
            [
                "gcloud", "filestore", "instances", "describe",
                config.instance_name,
                "--project", gke_config.project_id,
                "--zone", gke_config.zone,
                "--format=value(state)",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            state = result.stdout.strip()
            if state == "READY":
                logger.info("Filestore is ready")
                return True
            logger.debug(f"Filestore state: {state}")

        time.sleep(10)

    logger.error(f"Filestore did not become ready within {timeout} seconds")
    return False


def create_filestore_pv_pvc(
    config: FilestoreConfig, gke_config: GKEConfig, namespace: str
) -> bool:
    """Create PersistentVolume and PersistentVolumeClaim for Filestore.

    This creates a ReadWriteMany PV/PVC that can be mounted by multiple pods
    (gateway and workers) simultaneously.

    Args:
        config: Filestore configuration (must have ip_address set).
        gke_config: GKE configuration.
        namespace: Kubernetes namespace for the PVC.

    Returns:
        True if PV and PVC created successfully.
    """
    if not config.ip_address:
        logger.error("Filestore IP address not set. Call get_filestore_ip() first.")
        return False

    logger.info(f"Creating PV/PVC for Filestore at {config.ip_address}...")

    # Create namespace if it doesn't exist
    subprocess.run(
        ["kubectl", "create", "namespace", namespace],
        capture_output=True,
        text=True,
    )  # Ignore error if exists

    # PV and PVC manifest
    manifest = f"""---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: beta9-images-pv
spec:
  capacity:
    storage: {config.capacity_gb}Gi
  accessModes:
    - ReadWriteMany
  nfs:
    server: {config.ip_address}
    path: /{config.file_share_name}
  persistentVolumeReclaimPolicy: Delete
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: images
  namespace: {namespace}
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ""
  volumeName: beta9-images-pv
  resources:
    requests:
      storage: {config.capacity_gb}Gi
"""

    # Apply manifest via kubectl
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Failed to create PV/PVC: {result.stderr}")
        return False

    logger.info("PV/PVC created successfully (ReadWriteMany)")
    return True
