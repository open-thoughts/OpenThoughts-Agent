#!/usr/bin/env python3
"""
Beam Cluster Setup CLI

Main entry point for creating, managing, and validating a self-hosted
Beta9 cluster on GKE with Pinggy or LoadBalancer exposure.

Usage:
    python -m scripts.beam.cluster_setup create --project-id my-project --pinggy-url mybeam.a.pinggy.link --pinggy-token TOKEN
    python -m scripts.beam.cluster_setup destroy --project-id my-project
    python -m scripts.beam.cluster_setup status --project-id my-project
    python -m scripts.beam.cluster_setup validate --gateway-url https://mybeam.a.pinggy.link
"""

import argparse
import logging
import os
import sys

from scripts.beam.config import (
    Beta9Config,
    ClusterSetupConfig,
    FilestoreConfig,
    GKEConfig,
    LoadBalancerConfig,
    PinggyConfig,
    S3StorageConfig,
)
from scripts.beam.gke_provision import (
    check_gcloud_installed,
    check_git_installed,
    check_kubectl_installed,
    cluster_exists,
    create_cluster,
    delete_cluster,
    get_cluster_status,
    get_credentials,
    get_node_count,
    grant_cluster_admin,
)
from scripts.beam.s3_storage import (
    S3Storage,
    check_s3_configured,
    cleanup_juicefs_storage,
    init_s3_storage,
)
from scripts.beam.beta9_deploy import (
    add_helm_repo,
    check_helm_installed,
    cleanup_helm_repo,
    delete_namespace_pvcs,
    deploy_beta9,
    get_deployment_status,
    stop_log_collection,
    uninstall_beta9,
    wait_for_gateway_ready,
)
from scripts.beam.expose import (
    setup_loadbalancer_exposure,
    setup_pinggy_exposure,
    stop_exposure_processes,
    verify_endpoint_health,
)
from scripts.beam.validate_cluster import run_validation_suite


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def check_prerequisites() -> bool:
    """Check all required tools are installed."""
    all_ok = True

    if not check_gcloud_installed():
        all_ok = False

    if not check_kubectl_installed():
        all_ok = False

    if not check_helm_installed():
        all_ok = False

    if not check_git_installed():
        all_ok = False

    return all_ok


def cmd_create(args) -> int:
    """Create command: provision GKE, deploy Beta9, expose, validate."""
    logger = logging.getLogger(__name__)

    # Build configuration
    gke_config = GKEConfig(
        project_id=args.project_id,
        cluster_name=args.cluster_name,
        region=args.region,
        num_nodes=args.num_nodes,
        machine_type=args.machine_type,
    )

    beta9_config = Beta9Config(
        juicefs_fs_name=getattr(args, "juicefs_prefix", "beta9-fs"),
    )

    # Load S3 config for shared storage (optional for create command)
    s3_config = S3StorageConfig.from_env_optional()
    if s3_config:
        logger.info(f"Using external S3 storage: {s3_config.endpoint_url}")
    else:
        logger.warning("S3 storage not configured - using LocalStack (not recommended for production)")

    pinggy_config = None
    lb_config = None

    if args.expose_method == "pinggy":
        if not args.pinggy_url or not args.pinggy_token:
            logger.error("--pinggy-url and --pinggy-token required when expose-method=pinggy")
            return 1
        pinggy_config = PinggyConfig(
            persistent_url=args.pinggy_url,
            token=args.pinggy_token,
        )
    elif args.expose_method == "loadbalancer":
        lb_config = LoadBalancerConfig()

    # Check prerequisites
    logger.info("Checking prerequisites...")
    if not check_prerequisites():
        return 1

    # Step 1: Create GKE cluster
    if not args.skip_gke:
        if cluster_exists(gke_config):
            logger.info(f"Cluster '{gke_config.cluster_name}' already exists")
        else:
            if not create_cluster(gke_config, dry_run=args.dry_run):
                return 1

        # Get credentials
        if not get_credentials(gke_config, dry_run=args.dry_run):
            return 1

        # Grant cluster-admin (required for Helm charts with ClusterRoles)
        if not grant_cluster_admin(dry_run=args.dry_run):
            return 1
    else:
        logger.info("Skipping GKE provisioning (--skip-gke)")

    # Step 2: Deploy Beta9
    if not args.skip_beta9:
        if not add_helm_repo(dry_run=args.dry_run, config=beta9_config):
            return 1

        if not deploy_beta9(beta9_config, dry_run=args.dry_run, s3_config=s3_config):
            return 1

        if not args.dry_run:
            if not wait_for_gateway_ready(beta9_config):
                return 1
    else:
        logger.info("Skipping Beta9 deployment (--skip-beta9)")

    # Step 3: Expose cluster
    public_url = ""
    if not args.skip_expose:
        if args.expose_method == "pinggy":
            pf_proc, tunnel_proc, public_url = setup_pinggy_exposure(
                beta9_config, pinggy_config, dry_run=args.dry_run
            )
            if not args.dry_run and (pf_proc is None or tunnel_proc is None):
                logger.error("Failed to set up Pinggy exposure")
                return 1
        elif args.expose_method == "loadbalancer":
            public_url, success = setup_loadbalancer_exposure(
                beta9_config, lb_config, dry_run=args.dry_run
            )
            if not args.dry_run and not success:
                logger.error("Failed to set up LoadBalancer exposure")
                return 1

        # Verify endpoint
        if not args.dry_run and public_url:
            if not verify_endpoint_health(public_url):
                logger.warning("Endpoint health check failed (may need more time)")
    else:
        logger.info("Skipping exposure (--skip-expose)")

    # Step 4: Validate
    if not args.skip_validation and not args.dry_run and public_url:
        logger.info("Running validation tests...")
        # Use port 443 for Pinggy (TLS tunnel), 1993 for direct gRPC access
        gateway_port = 443 if args.expose_method == "pinggy" else 1993
        report = run_validation_suite(public_url, num_lifecycle_tests=2, gateway_port=gateway_port)
        if report.failed > 0:
            logger.warning(f"Validation: {report.failed} tests failed")
    else:
        logger.info("Skipping validation (--skip-validation or dry-run)")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BEAM CLUSTER SETUP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Project:      {gke_config.project_id}")
    logger.info(f"  Cluster:      {gke_config.cluster_name}")
    logger.info(f"  Region:       {gke_config.region}")
    logger.info(f"  Nodes:        {gke_config.num_nodes} x {gke_config.machine_type}")
    if public_url:
        logger.info(f"  Public URL:   {public_url}")
        logger.info("")
        logger.info("To use with Harbor/beta9 SDK, set:")
        logger.info(f"  export BETA9_GATEWAY_HOST={public_url.replace('https://', '').replace('http://', '')}")
    logger.info("=" * 60)

    if args.expose_method == "pinggy" and not args.dry_run:
        logger.info("")
        logger.info("NOTE: Pinggy tunnel is running in background.")
        logger.info("      Press Ctrl+C to stop when done.")
        try:
            # Keep script running to maintain tunnel
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            stop_exposure_processes()

    return 0


def cmd_destroy(args) -> int:
    """Destroy command: tear down everything."""
    logger = logging.getLogger(__name__)

    gke_config = GKEConfig(
        project_id=args.project_id,
        cluster_name=args.cluster_name,
        region=args.region,
    )

    beta9_config = Beta9Config(
        juicefs_fs_name=getattr(args, "juicefs_prefix", "beta9-fs"),
    )

    # Step 1: Stop exposure processes
    logger.info("Stopping exposure processes...")
    stop_exposure_processes()

    # Step 2: Uninstall Beta9 and delete PVCs
    if not args.skip_beta9:
        # Get credentials first (may fail if cluster doesn't exist)
        get_credentials(gke_config, dry_run=args.dry_run)
        uninstall_beta9(beta9_config, dry_run=args.dry_run)
        # Delete PVCs to clean up GCP persistent disks
        # (Helm uninstall doesn't delete PVCs by default)
        delete_namespace_pvcs(beta9_config.namespace, dry_run=args.dry_run)
    else:
        logger.info("Skipping Beta9 uninstall (--skip-beta9)")

    # Step 3: Delete GKE cluster
    if not args.skip_gke:
        if not delete_cluster(gke_config, dry_run=args.dry_run):
            return 1
    else:
        logger.info("Skipping GKE deletion (--skip-gke)")

    logger.info("Cluster destroyed successfully")
    return 0


def cmd_status(args) -> int:
    """Status command: show current cluster status."""
    logger = logging.getLogger(__name__)

    gke_config = GKEConfig(
        project_id=args.project_id,
        cluster_name=args.cluster_name,
        region=args.region,
    )

    beta9_config = Beta9Config()

    logger.info("=" * 60)
    logger.info("BEAM CLUSTER STATUS")
    logger.info("=" * 60)

    # GKE status
    status = get_cluster_status(gke_config)
    if status:
        logger.info(f"  GKE Cluster:    {gke_config.cluster_name} ({status})")

        # Get credentials and check nodes
        if get_credentials(gke_config, dry_run=False):
            node_count = get_node_count(gke_config)
            logger.info(f"  Nodes:          {node_count}")
    else:
        logger.info(f"  GKE Cluster:    {gke_config.cluster_name} (NOT FOUND)")
        return 0

    # Beta9 status
    deployment_status = get_deployment_status(beta9_config)
    logger.info(f"  Beta9 Installed: {deployment_status['installed']}")
    logger.info(f"  Gateway Ready:   {deployment_status['gateway_ready']}")
    logger.info(f"  Worker Ready:    {deployment_status['worker_ready']}")

    if deployment_status['pods']:
        logger.info("  Pods:")
        for pod in deployment_status['pods']:
            ready = "Ready" if pod['ready'] else "Not Ready"
            logger.info(f"    - {pod['name']}: {pod['phase']} ({ready})")

    logger.info("=" * 60)
    return 0


def cmd_validate(args) -> int:
    """Validate command: run health checks on existing cluster."""
    logger = logging.getLogger(__name__)

    if not args.gateway_url:
        logger.error("--gateway-url required for validate command")
        return 1

    # Use port 443 for Pinggy (TLS tunnel), 1993 for direct gRPC access
    gateway_port = getattr(args, 'gateway_port', 443)

    report = run_validation_suite(
        args.gateway_url,
        num_lifecycle_tests=args.num_tests,
        include_isolation_test=not args.skip_isolation,
        gateway_port=gateway_port,
    )

    return 0 if report.failed == 0 else 1


def cmd_test(args) -> int:
    """Test command: create cluster, validate, then tear down automatically."""
    logger = logging.getLogger(__name__)

    # Build configuration
    gke_config = GKEConfig(
        project_id=args.project_id,
        cluster_name=args.cluster_name,
        region=args.region,
        num_nodes=args.num_nodes,
        machine_type=args.machine_type,
    )

    beta9_config = Beta9Config(
        juicefs_fs_name=getattr(args, "juicefs_prefix", "beta9-fs"),
    )

    # Initialize S3 storage for artifact caching (optional but recommended)
    s3_storage = None
    s3_config = S3StorageConfig.from_env_optional()

    pinggy_config = None
    lb_config = None

    if args.expose_method == "pinggy":
        if not args.pinggy_url or not args.pinggy_token:
            logger.error("--pinggy-url and --pinggy-token required when expose-method=pinggy")
            return 1
        pinggy_config = PinggyConfig(
            persistent_url=args.pinggy_url,
            token=args.pinggy_token,
        )
    elif args.expose_method == "loadbalancer":
        lb_config = LoadBalancerConfig()

    validation_passed = False
    public_url = ""
    resources_created = False  # Track if we actually created anything

    # Check prerequisites BEFORE the try block
    logger.info("=" * 60)
    logger.info("BEAM CLUSTER TEST: Setup -> Validate -> Teardown")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Checking prerequisites...")
    if not check_prerequisites():
        return 1

    try:

        # Step 1: Create GKE cluster
        logger.info("")
        logger.info("[1/5] Creating GKE cluster...")
        if cluster_exists(gke_config):
            logger.info(f"Cluster '{gke_config.cluster_name}' already exists")
            resources_created = True
        else:
            if not create_cluster(gke_config, dry_run=args.dry_run):
                return 1
            resources_created = True

        # Get credentials
        if not get_credentials(gke_config, dry_run=args.dry_run):
            return 1

        # Grant cluster-admin (required for Helm charts with ClusterRoles)
        if not grant_cluster_admin(dry_run=args.dry_run):
            return 1

        # Step 2: Initialize S3 storage for artifact caching (REQUIRED)
        logger.info("")
        logger.info("[2/5] Initializing S3 storage for artifact caching...")
        if not s3_config:
            logger.error("S3 storage not configured!")
            logger.error("Required environment variables: LAION_BUCKET_NAME, LAION_ACCESS_KEY, LAION_SECRET_KEY, LAION_ENDPOINT")
            logger.error("Please source your secrets.env file before running this command.")
            raise RuntimeError("S3 storage configuration required but not found")

        s3_storage = S3Storage(s3_config)
        if not args.dry_run:
            if not s3_storage.ensure_namespace_exists():
                logger.error(f"Failed to initialize S3 storage at s3://{s3_config.bucket_name}/{s3_config.namespace}/")
                raise RuntimeError("S3 storage initialization failed")

            logger.info(f"S3 storage ready: s3://{s3_config.bucket_name}/{s3_config.namespace}/")

            # Clean up JuiceFS storage from previous deployments
            # JuiceFS refuses to format if the storage path is not empty
            logger.info(f"Cleaning up JuiceFS storage ({beta9_config.juicefs_fs_name}) from previous deployments...")
            if not cleanup_juicefs_storage(
                bucket_name=s3_config.bucket_name,
                juicefs_prefix=beta9_config.juicefs_fs_name,
            ):
                logger.warning("JuiceFS cleanup failed - deployment may fail if storage is not empty")
            # List existing cached artifacts
            artifacts = s3_storage.list_artifacts()
            if artifacts:
                logger.info(f"Found {len(artifacts)} cached artifacts available for reuse")
                for artifact in artifacts[:5]:  # Show first 5
                    logger.info(f"  - {artifact.get('image_name', 'unknown')}")
                if len(artifacts) > 5:
                    logger.info(f"  ... and {len(artifacts) - 5} more")
        else:
            logger.info(f"[DRY-RUN] Would initialize S3 storage: s3://{s3_config.bucket_name}/{s3_config.namespace}/")

        # Step 3: Deploy Beta9
        logger.info("")
        logger.info("[3/5] Deploying Beta9...")
        if not add_helm_repo(dry_run=args.dry_run, config=beta9_config):
            return 1

        if not deploy_beta9(beta9_config, dry_run=args.dry_run, s3_config=s3_config):
            return 1

        if not args.dry_run:
            if not wait_for_gateway_ready(beta9_config):
                return 1

        # Step 4: Expose cluster
        logger.info("")
        logger.info("[4/5] Exposing cluster...")
        if args.expose_method == "pinggy":
            pf_proc, tunnel_proc, public_url = setup_pinggy_exposure(
                beta9_config, pinggy_config, dry_run=args.dry_run
            )
            if not args.dry_run and (pf_proc is None or tunnel_proc is None):
                logger.error("Failed to set up Pinggy exposure")
                return 1
        elif args.expose_method == "loadbalancer":
            public_url, success = setup_loadbalancer_exposure(
                beta9_config, lb_config, dry_run=args.dry_run
            )
            if not args.dry_run and not success:
                logger.error("Failed to set up LoadBalancer exposure")
                return 1

        # Verify endpoint
        if not args.dry_run and public_url:
            if not verify_endpoint_health(public_url):
                logger.warning("Endpoint health check failed (may need more time)")

        # Step 5: Run validation
        logger.info("")
        logger.info("[5/5] Running validation tests...")
        if not args.dry_run and public_url:
            # Use port 443 for Pinggy (TLS tunnel), 1993 for direct gRPC access
            gateway_port = 443 if args.expose_method == "pinggy" else 1993
            report = run_validation_suite(
                public_url,
                num_lifecycle_tests=args.num_tests,
                include_isolation_test=not args.skip_isolation,
                gateway_port=gateway_port,
            )
            validation_passed = report.failed == 0
        else:
            logger.info("Skipping validation (dry-run)")
            validation_passed = True

    finally:
        # Stop log collection first (saves logs before cleanup)
        stop_log_collection()

        # Only tear down if we actually created resources
        if not resources_created:
            logger.info("No resources were created, skipping teardown")
        else:
            logger.info("")
            logger.info("=" * 60)
            logger.info("TEARDOWN: Cleaning up resources...")
            logger.info("=" * 60)

            # Stop exposure processes
            logger.info("Stopping exposure processes...")
            stop_exposure_processes()

            # Uninstall Beta9 and delete PVCs
            get_credentials(gke_config, dry_run=args.dry_run)
            uninstall_beta9(beta9_config, dry_run=args.dry_run)
            # Delete PVCs to clean up GCP persistent disks
            delete_namespace_pvcs(beta9_config.namespace, dry_run=args.dry_run)

            # Note: S3 storage artifacts are NOT deleted - they are cached for reuse
            if s3_storage:
                logger.info("S3 cached artifacts preserved for future reuse")

            # Delete GKE cluster
            if not args.keep_cluster:
                delete_cluster(gke_config, dry_run=args.dry_run)
            else:
                logger.info("Keeping cluster (--keep-cluster specified)")

        # Always clean up temp helm repo directory
        cleanup_helm_repo()

    # Print final result
    logger.info("")
    logger.info("=" * 60)
    if validation_passed:
        logger.info("TEST RESULT: PASSED")
    else:
        logger.info("TEST RESULT: FAILED")
    logger.info("=" * 60)

    return 0 if validation_passed else 1


def main():
    parser = argparse.ArgumentParser(
        description="Beam Cluster Setup CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create cluster with Pinggy tunnel (interactive, keeps tunnel running)
  python -m scripts.beam.cluster_setup create \\
    --project-id my-gcp-project \\
    --num-nodes 8 \\
    --pinggy-url mybeam.a.pinggy.link \\
    --pinggy-token $PINGGY_TOKEN

  # Create cluster with LoadBalancer (non-interactive)
  python -m scripts.beam.cluster_setup create \\
    --project-id my-gcp-project \\
    --expose-method loadbalancer

  # Dry run to preview commands
  python -m scripts.beam.cluster_setup create \\
    --project-id my-gcp-project \\
    --dry-run

  # Tear down everything
  python -m scripts.beam.cluster_setup destroy --project-id my-gcp-project

  # Run validation tests only
  python -m scripts.beam.cluster_setup validate \\
    --gateway-url https://mybeam.a.pinggy.link

  # Full test cycle: create -> validate -> teardown
  python -m scripts.beam.cluster_setup test \\
    --project-id my-gcp-project \\
    --num-nodes 2
        """,
    )

    # Global options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create GKE cluster and deploy Beta9")
    create_parser.add_argument("--project-id", required=True, help="GCP project ID")
    create_parser.add_argument("--cluster-name", default="beam-sandbox-cluster", help="GKE cluster name")
    create_parser.add_argument("--region", default="us-central1", help="GCP region")
    create_parser.add_argument("--num-nodes", type=int, default=8, help="Number of CPU nodes")
    create_parser.add_argument("--machine-type", default="e2-standard-4", help="GCE machine type")
    create_parser.add_argument("--expose-method", choices=["pinggy", "loadbalancer"], default="pinggy",
                               help="How to expose cluster")
    create_parser.add_argument("--pinggy-url", help="Pinggy persistent URL")
    create_parser.add_argument("--pinggy-token", help="Pinggy auth token")
    create_parser.add_argument("--skip-gke", action="store_true", help="Skip GKE provisioning")
    create_parser.add_argument("--skip-beta9", action="store_true", help="Skip Beta9 deployment")
    create_parser.add_argument("--skip-expose", action="store_true", help="Skip internet exposure")
    create_parser.add_argument("--skip-validation", action="store_true", help="Skip validation tests")
    create_parser.add_argument("--juicefs-prefix", default="beta9-fs",
                               help="JuiceFS S3 prefix (change for multiple clusters)")
    create_parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    # Destroy command
    destroy_parser = subparsers.add_parser("destroy", help="Tear down cluster and Beta9")
    destroy_parser.add_argument("--project-id", required=True, help="GCP project ID")
    destroy_parser.add_argument("--cluster-name", default="beam-sandbox-cluster", help="GKE cluster name")
    destroy_parser.add_argument("--region", default="us-central1", help="GCP region")
    destroy_parser.add_argument("--skip-gke", action="store_true", help="Skip GKE deletion")
    destroy_parser.add_argument("--skip-beta9", action="store_true", help="Skip Beta9 uninstall")
    destroy_parser.add_argument("--juicefs-prefix", default="beta9-fs",
                               help="JuiceFS S3 prefix (for cleaning up storage)")
    destroy_parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show cluster status")
    status_parser.add_argument("--project-id", required=True, help="GCP project ID")
    status_parser.add_argument("--cluster-name", default="beam-sandbox-cluster", help="GKE cluster name")
    status_parser.add_argument("--region", default="us-central1", help="GCP region")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Run validation tests")
    validate_parser.add_argument("--gateway-url", help="Beta9 gateway URL")
    validate_parser.add_argument("--gateway-port", type=int, default=443,
                                 help="Gateway gRPC port (443 for Pinggy tunnels, 1993 for direct)")
    validate_parser.add_argument("--num-tests", type=int, default=3, help="Number of lifecycle tests")
    validate_parser.add_argument("--skip-isolation", action="store_true", help="Skip isolation test")

    # Test command (create -> validate -> destroy)
    test_parser = subparsers.add_parser("test", help="Create cluster, validate, then tear down")
    test_parser.add_argument("--project-id", required=True, help="GCP project ID")
    test_parser.add_argument("--cluster-name", default="beam-test-cluster", help="GKE cluster name")
    test_parser.add_argument("--region", default="us-central1", help="GCP region")
    test_parser.add_argument("--num-nodes", type=int, default=2, help="Number of CPU nodes (default: 2 for testing)")
    test_parser.add_argument("--machine-type", default="e2-standard-4", help="GCE machine type")
    test_parser.add_argument("--expose-method", choices=["pinggy", "loadbalancer"], default="loadbalancer",
                             help="How to expose cluster (default: loadbalancer for testing)")
    test_parser.add_argument("--pinggy-url", help="Pinggy persistent URL")
    test_parser.add_argument("--pinggy-token", help="Pinggy auth token")
    test_parser.add_argument("--num-tests", type=int, default=3, help="Number of validation tests")
    test_parser.add_argument("--skip-isolation", action="store_true", help="Skip isolation test")
    test_parser.add_argument("--keep-cluster", action="store_true", help="Don't delete cluster after test")
    test_parser.add_argument("--juicefs-prefix", default="beta9-fs",
                               help="JuiceFS S3 prefix (change for multiple clusters)")
    test_parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    if args.command == "create":
        return cmd_create(args)
    elif args.command == "destroy":
        return cmd_destroy(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "test":
        return cmd_test(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
