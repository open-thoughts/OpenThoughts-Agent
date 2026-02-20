"""Sandbox health check validation for Beta9 cluster."""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Module-level token cache to avoid re-fetching on each test
_cached_token: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of a single validation test."""

    test_name: str
    passed: bool
    duration_sec: float
    error: Optional[str] = None
    stdout: Optional[str] = None


@dataclass
class ValidationReport:
    """Full validation report."""

    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests

    def add_result(self, result: ValidationResult):
        self.results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        lines = [
            f"Validation Report: {self.passed}/{self.total_tests} passed ({self.success_rate:.0%})",
            "-" * 50,
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.test_name} ({r.duration_sec:.2f}s)")
            if r.error:
                lines.append(f"         Error: {r.error}")
        return "\n".join(lines)


def get_or_create_token(gateway_host: str, gateway_port: int = 1993, force_refresh: bool = False) -> Optional[str]:
    """Get an auth token from the Beta9 gateway.

    On a fresh gateway, calling Authorize without a token will:
    1. Create a new workspace
    2. Return a new cluster admin token

    Tokens are cached module-level to avoid re-fetching on each test.

    Args:
        gateway_host: Gateway hostname (without port).
        gateway_port: Gateway gRPC port.
        force_refresh: If True, ignore cached token and fetch new one.

    Returns:
        Auth token string, or None if failed.
    """
    global _cached_token

    # Return cached token if available
    if _cached_token and not force_refresh:
        logger.debug(f"Using cached token")
        return _cached_token

    import grpc

    try:
        # Import from beta9 SDK (classes are in clients.gateway, not proto)
        from beta9.clients.gateway import AuthorizeRequest, GatewayServiceStub
    except ImportError:
        logger.warning("beta9 SDK not installed, cannot get token")
        return None

    channel = None
    try:
        # Connect to gateway (insecure for local/direct connections)
        # Add connection timeout
        channel = grpc.insecure_channel(
            f"{gateway_host}:{gateway_port}",
            options=[
                ('grpc.connect_timeout_ms', 10000),
                ('grpc.keepalive_time_ms', 30000),
            ]
        )
        stub = GatewayServiceStub(channel)

        # Call authorize - this doesn't require auth and will create workspace/token on fresh gateway
        logger.info(f"Calling authorize on {gateway_host}:{gateway_port}...")
        response = stub.authorize(AuthorizeRequest())

        if response.ok:
            if response.new_token:
                logger.info(f"Got new token from gateway (workspace: {response.workspace_id})")
                _cached_token = response.new_token
                return response.new_token
            else:
                logger.info(f"Authorized with existing workspace: {response.workspace_id}")
                # Gateway already has a workspace but didn't give us a token
                # This means we need to use an existing token
                logger.warning("Gateway has existing workspace but no token returned - need existing token")
                return None
        else:
            logger.warning(f"Authorization failed: {response.error_msg}")
            return None

    except grpc.RpcError as e:
        logger.warning(f"gRPC error getting token: {e.code()} - {e.details()}")
        return None
    except Exception as e:
        logger.warning(f"Failed to get token from gateway: {e}")
        return None
    finally:
        if channel:
            try:
                channel.close()
            except Exception:
                pass


def configure_beta9_endpoint(gateway_url: str, gateway_port: int = 1993, token: Optional[str] = None):
    """Configure beta9 SDK to use custom gateway endpoint.

    This properly saves the config to ~/.beta9/ so the SDK uses the correct
    gateway host and port for gRPC connections.

    If no token is provided, attempts to get one from the gateway's Authorize endpoint.

    Args:
        gateway_url: Full URL to Beta9 gateway (e.g., https://mybeam.a.pinggy.link).
        gateway_port: Gateway gRPC port (default 1993, or 443 for Pinggy tunnels).
        token: Auth token for Beta9 gateway (if None, will try to get from gateway).
    """
    from pathlib import Path
    from urllib.parse import urlparse

    try:
        from beta9.config import ConfigContext, save_config
    except ImportError:
        logger.warning("beta9 SDK not installed, skipping config save")
        return

    # Extract hostname from URL (strip any port - we use gateway_port for gRPC)
    parsed = urlparse(gateway_url if "://" in gateway_url else f"http://{gateway_url}")
    gateway_host = parsed.hostname or gateway_url.replace("https://", "").replace("http://", "").split(":")[0]

    # If no token provided, try to get one from the gateway
    if not token:
        token = get_or_create_token(gateway_host, gateway_port)
        if not token:
            logger.warning("No token available - authentication will likely fail")
            token = "no-token-available"

    # Create config context
    ctx = ConfigContext(
        token=token,
        gateway_host=gateway_host,
        gateway_port=gateway_port,
    )

    # Create .beta9 directory if needed
    config_dir = Path.home() / ".beta9"
    config_dir.mkdir(exist_ok=True)

    # Save config (this is what the SDK actually reads)
    save_config({"default": ctx})

    # Also set environment variables as backup
    os.environ["BETA9_GATEWAY_HOST"] = gateway_host
    os.environ["BETA9_API_URL"] = gateway_url.rstrip("/")

    logger.info(f"Configured beta9 endpoint: {gateway_host}:{gateway_port}")


def _is_cold_start_error(error: str) -> bool:
    """Check if an error is a typical cold-start/transient error.

    Cold-start errors occur when:
    - First image build hasn't completed yet
    - Container infrastructure is still warming up
    - Race conditions in container scheduling

    Args:
        error: Error message string.

    Returns:
        True if error appears to be a cold-start issue.
    """
    cold_start_patterns = [
        "UnknownError",
        "container not found",
        "image not found",
        "build failed",
        "timeout",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "connection refused",
        "no such container",
        "ContainerCreating",
    ]
    error_lower = error.lower()
    return any(pattern.lower() in error_lower for pattern in cold_start_patterns)


def validate_sandbox_lifecycle(
    gateway_url: str,
    test_id: str = None,
    gateway_port: int = 443,
    max_retries: int = 0,
    retry_delay_sec: int = 5,
) -> ValidationResult:
    """Test sandbox create -> exec -> terminate lifecycle.

    Args:
        gateway_url: Beta9 gateway URL.
        test_id: Optional test identifier.
        gateway_port: Gateway gRPC port (443 for Pinggy tunnels, 1993 for direct).
        max_retries: Maximum number of retries on cold-start errors (0 = no retries).
        retry_delay_sec: Delay between retries in seconds.

    Returns:
        ValidationResult with test outcome.
    """
    test_name = f"sandbox_lifecycle_{test_id or uuid4().hex[:6]}"
    start_time = time.time()
    last_error = None

    for attempt in range(max_retries + 1):
        instance = None
        stdout = None

        if attempt > 0:
            logger.info(f"[{test_name}] Retry {attempt}/{max_retries} after cold-start error...")
            time.sleep(retry_delay_sec)

        try:
            # Import beta9 SDK
            from beta9 import Image, PythonVersion, Sandbox

            # Configure endpoint with correct port for Pinggy (443) or direct (1993)
            configure_beta9_endpoint(gateway_url, gateway_port=gateway_port)

            # Create sandbox
            logger.info(f"[{test_name}] Creating sandbox...")
            sandbox = Sandbox(
                name=f"health-check-{uuid4().hex[:8]}",
                image=Image(python_version=PythonVersion.Python311),
                cpu=1,
                memory=512,
                env={"TEST_VAR": "beam-health-check"},
            )

            instance = sandbox.create()
            logger.info(f"[{test_name}] Sandbox created")

            # Execute test command
            logger.info(f"[{test_name}] Executing test command...")
            test_string = f"hello-beam-{uuid4().hex[:6]}"

            # Use async execution
            async def run_command():
                process = await instance.aio.process.exec("echo", test_string)
                exit_code = await process.wait()
                stdout = process._sync.stdout.read()
                return exit_code, stdout

            exit_code, stdout = asyncio.run(run_command())

            # Verify output
            if exit_code != 0:
                raise RuntimeError(f"Command exited with code {exit_code}")

            if test_string not in stdout:
                raise RuntimeError(f"Expected '{test_string}' in output, got: {stdout}")

            logger.info(f"[{test_name}] Command executed successfully")

            duration = time.time() - start_time
            return ValidationResult(
                test_name=test_name,
                passed=True,
                duration_sec=duration,
                stdout=stdout,
            )

        except ImportError as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name=test_name,
                passed=False,
                duration_sec=duration,
                error=f"beta9 SDK not installed: {e}",
            )
        except Exception as e:
            last_error = str(e)

            # Check if this is a cold-start error that might resolve with retry
            if attempt < max_retries and _is_cold_start_error(last_error):
                logger.warning(f"[{test_name}] Cold-start error (attempt {attempt + 1}): {last_error}")
                # Continue to retry
            else:
                # Either no retries left or not a cold-start error
                duration = time.time() - start_time
                return ValidationResult(
                    test_name=test_name,
                    passed=False,
                    duration_sec=duration,
                    error=last_error,
                )
        finally:
            # Always clean up the sandbox
            if instance is not None:
                try:
                    logger.info(f"[{test_name}] Terminating sandbox...")
                    instance.terminate()
                    logger.info(f"[{test_name}] Sandbox terminated")
                except Exception as cleanup_error:
                    logger.warning(f"[{test_name}] Failed to terminate sandbox: {cleanup_error}")

    # Should not reach here, but handle edge case
    duration = time.time() - start_time
    return ValidationResult(
        test_name=test_name,
        passed=False,
        duration_sec=duration,
        error=last_error or "Unknown error after retries",
    )


def validate_sandbox_isolation(
    gateway_url: str,
    gateway_port: int = 443,
    max_retries: int = 1,
    retry_delay_sec: int = 5,
) -> ValidationResult:
    """Test that sandboxes are properly isolated.

    Creates two sandboxes and verifies they cannot see each other's files.

    Args:
        gateway_url: Beta9 gateway URL.
        gateway_port: Gateway gRPC port (443 for Pinggy tunnels, 1993 for direct).
        max_retries: Maximum number of retries on cold-start errors.
        retry_delay_sec: Delay between retries in seconds.

    Returns:
        ValidationResult with test outcome.
    """
    test_name = "sandbox_isolation"
    start_time = time.time()
    last_error = None

    for attempt in range(max_retries + 1):
        instance1 = None
        instance2 = None

        if attempt > 0:
            logger.info(f"[{test_name}] Retry {attempt}/{max_retries} after cold-start error...")
            time.sleep(retry_delay_sec)

        try:
            from beta9 import Image, PythonVersion, Sandbox

            configure_beta9_endpoint(gateway_url, gateway_port=gateway_port)

            # Create first sandbox and write a file
            sandbox1 = Sandbox(
                name=f"isolation-test-1-{uuid4().hex[:6]}",
                image=Image(python_version=PythonVersion.Python311),
                cpu=1,
                memory=512,
            )
            instance1 = sandbox1.create()

            secret_value = f"secret-{uuid4().hex}"

            async def write_file():
                process = await instance1.aio.process.exec(
                    "bash", "-c", f"echo '{secret_value}' > /tmp/secret.txt"
                )
                await process.wait()

            asyncio.run(write_file())

            # Create second sandbox and try to read the file
            sandbox2 = Sandbox(
                name=f"isolation-test-2-{uuid4().hex[:6]}",
                image=Image(python_version=PythonVersion.Python311),
                cpu=1,
                memory=512,
            )
            instance2 = sandbox2.create()

            async def read_file():
                process = await instance2.aio.process.exec(
                    "bash", "-c", "cat /tmp/secret.txt 2>/dev/null || echo 'FILE_NOT_FOUND'"
                )
                exit_code = await process.wait()
                stdout = process._sync.stdout.read()
                return stdout

            stdout = asyncio.run(read_file())

            # Verify isolation
            if secret_value in stdout:
                raise RuntimeError("Sandbox isolation failed: secret visible across sandboxes")

            duration = time.time() - start_time
            return ValidationResult(
                test_name=test_name,
                passed=True,
                duration_sec=duration,
            )

        except ImportError as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name=test_name,
                passed=False,
                duration_sec=duration,
                error=f"beta9 SDK not installed: {e}",
            )
        except Exception as e:
            last_error = str(e)

            # Check if this is a cold-start error that might resolve with retry
            if attempt < max_retries and _is_cold_start_error(last_error):
                logger.warning(f"[{test_name}] Cold-start error (attempt {attempt + 1}): {last_error}")
                # Continue to retry
            else:
                duration = time.time() - start_time
                return ValidationResult(
                    test_name=test_name,
                    passed=False,
                    duration_sec=duration,
                    error=last_error,
                )
        finally:
            # Always clean up sandboxes
            for idx, instance in enumerate([instance1, instance2], start=1):
                if instance is not None:
                    try:
                        logger.info(f"[{test_name}] Terminating sandbox {idx}...")
                        instance.terminate()
                        logger.info(f"[{test_name}] Sandbox {idx} terminated")
                    except Exception as cleanup_error:
                        logger.warning(f"[{test_name}] Failed to terminate sandbox {idx}: {cleanup_error}")

    # Should not reach here, but handle edge case
    duration = time.time() - start_time
    return ValidationResult(
        test_name=test_name,
        passed=False,
        duration_sec=duration,
        error=last_error or "Unknown error after retries",
    )


def wait_for_grpc_ready(
    gateway_host: str,
    gateway_port: int = 1993,
    max_attempts: int = 6,
    delay_sec: int = 10,
) -> bool:
    """Wait for gRPC endpoint to be ready by attempting to get a token.

    Args:
        gateway_host: Gateway hostname.
        gateway_port: Gateway gRPC port.
        max_attempts: Maximum number of connection attempts.
        delay_sec: Delay between attempts in seconds.

    Returns:
        True if endpoint is ready, False otherwise.
    """
    global _cached_token

    for attempt in range(1, max_attempts + 1):
        logger.info(f"Checking gRPC endpoint readiness (attempt {attempt}/{max_attempts})...")

        # If we have a cached token from bootstrap (via port-forward), verify
        # connectivity to the external endpoint using that token
        if _cached_token:
            if _verify_grpc_connectivity(gateway_host, gateway_port, _cached_token):
                logger.info("gRPC endpoint is ready (verified with cached token)")
                return True
        else:
            # No cached token - try to get a new one (fresh deployment)
            token = get_or_create_token(gateway_host, gateway_port, force_refresh=True)
            if token:
                logger.info("gRPC endpoint is ready and token obtained")
                _cached_token = token
                return True

        if attempt < max_attempts:
            logger.info(f"Endpoint not ready, waiting {delay_sec}s before retry...")
            time.sleep(delay_sec)

    logger.warning(f"gRPC endpoint not ready after {max_attempts} attempts")
    return False


def _verify_grpc_connectivity(gateway_host: str, gateway_port: int, token: str) -> bool:
    """Verify gRPC connectivity using an existing token.

    Makes a lightweight gRPC call to verify the endpoint is reachable.

    Args:
        gateway_host: Gateway hostname.
        gateway_port: Gateway gRPC port.
        token: Auth token to use.

    Returns:
        True if endpoint is reachable, False otherwise.
    """
    import grpc

    try:
        from beta9.clients.gateway import GatewayServiceStub, AuthorizeRequest
    except ImportError as e:
        logger.warning(f"beta9 SDK import failed: {e}")
        return False

    channel = None
    try:
        logger.info(f"Verifying gRPC connectivity to {gateway_host}:{gateway_port}...")
        channel = grpc.insecure_channel(
            f"{gateway_host}:{gateway_port}",
            options=[
                ('grpc.connect_timeout_ms', 10000),
                ('grpc.keepalive_time_ms', 30000),
            ]
        )
        stub = GatewayServiceStub(channel)

        # Use Authorize to verify connectivity - same endpoint as bootstrap
        # ANY response (even error) proves the gateway is reachable and responding
        # Only gRPC exceptions (timeout, connection refused) mean no connectivity
        response = stub.authorize(AuthorizeRequest())

        # Got a response - connectivity confirmed!
        if response.ok:
            logger.info(f"Connectivity verified (workspace: {response.workspace_id})")
        else:
            # Error response still proves connectivity (gateway responded)
            logger.info(f"Connectivity verified (gateway responded: {response.error_msg})")
        return True

    except grpc.RpcError as e:
        logger.warning(f"gRPC connectivity check failed: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        logger.warning(f"Connectivity check failed: {e}")
        return False
    finally:
        if channel:
            try:
                channel.close()
            except Exception:
                pass


def _run_warmup_sandbox(gateway_url: str, gateway_port: int = 443, max_warmup_attempts: int = 3) -> bool:
    """Run a warm-up sandbox to prime image cache before validation tests.

    On a cold Beta9 cluster, the first sandbox creation triggers an image build
    which can take time and may fail. This warm-up phase attempts to prime the
    cache so subsequent tests run reliably.

    Args:
        gateway_url: Beta9 gateway URL.
        gateway_port: Gateway gRPC port.
        max_warmup_attempts: Maximum attempts to warm up the cluster.

    Returns:
        True if warm-up succeeded (cluster is ready), False otherwise.
    """
    logger.info("Running warm-up phase to prime image cache...")

    for attempt in range(1, max_warmup_attempts + 1):
        logger.info(f"Warm-up attempt {attempt}/{max_warmup_attempts}...")

        # Run a lifecycle test with retries - we expect early attempts might fail
        result = validate_sandbox_lifecycle(
            gateway_url,
            test_id=f"warmup_{attempt}",
            gateway_port=gateway_port,
            max_retries=2,  # Allow retries within each warm-up attempt
            retry_delay_sec=10,
        )

        if result.passed:
            logger.info(f"Warm-up succeeded on attempt {attempt} ({result.duration_sec:.1f}s)")
            return True

        # Check if this looks like a cold-start error we might recover from
        if result.error and _is_cold_start_error(result.error):
            logger.warning(f"Warm-up attempt {attempt} failed with cold-start error: {result.error}")
            if attempt < max_warmup_attempts:
                # Wait longer between warm-up attempts to let image build complete
                wait_time = 15 * attempt  # Progressive backoff: 15s, 30s, 45s
                logger.info(f"Waiting {wait_time}s for image build to complete...")
                time.sleep(wait_time)
        else:
            # Non-transient error - don't keep trying
            logger.error(f"Warm-up failed with non-transient error: {result.error}")
            return False

    logger.warning(f"Warm-up failed after {max_warmup_attempts} attempts")
    return False


def run_validation_suite(
    gateway_url: str,
    num_lifecycle_tests: int = 3,
    include_isolation_test: bool = True,
    gateway_port: int = 443,
    wait_for_ready: bool = True,
    warmup_cluster: bool = True,
    max_warmup_attempts: int = 3,
) -> ValidationReport:
    """Run full validation suite.

    Args:
        gateway_url: Beta9 gateway URL.
        num_lifecycle_tests: Number of sandbox lifecycle tests to run.
        include_isolation_test: Whether to include isolation test.
        gateway_port: Gateway gRPC port (443 for Pinggy tunnels, 1993 for direct).
        wait_for_ready: Whether to wait for gRPC endpoint to be ready before tests.
        warmup_cluster: Whether to run warm-up phase for cold clusters.
        max_warmup_attempts: Maximum attempts for warm-up phase.

    Returns:
        ValidationReport with all test results.
    """
    from urllib.parse import urlparse

    report = ValidationReport()

    logger.info(f"Running validation suite against: {gateway_url}")
    logger.info(f"  Lifecycle tests: {num_lifecycle_tests}")
    logger.info(f"  Isolation test: {include_isolation_test}")
    logger.info(f"  Gateway port: {gateway_port}")
    logger.info(f"  Warm-up enabled: {warmup_cluster}")

    # Extract hostname for gRPC connection
    parsed = urlparse(gateway_url if "://" in gateway_url else f"http://{gateway_url}")
    gateway_host = parsed.hostname or gateway_url.replace("https://", "").replace("http://", "").split(":")[0]

    # Wait for gRPC endpoint to be ready (important for fresh LoadBalancer)
    if wait_for_ready:
        if not wait_for_grpc_ready(gateway_host, gateway_port):
            # Add a failed result for connectivity
            report.add_result(ValidationResult(
                test_name="grpc_connectivity",
                passed=False,
                duration_sec=0,
                error="gRPC endpoint not reachable after retries",
            ))
            return report

    # Run warm-up phase to prime image cache on cold clusters
    if warmup_cluster:
        warmup_start = time.time()
        warmup_success = _run_warmup_sandbox(gateway_url, gateway_port, max_warmup_attempts)
        warmup_duration = time.time() - warmup_start

        # Record warm-up result (informational - doesn't fail the suite)
        report.add_result(ValidationResult(
            test_name="cluster_warmup",
            passed=warmup_success,
            duration_sec=warmup_duration,
            error=None if warmup_success else "Warm-up phase did not fully succeed",
        ))

        if not warmup_success:
            logger.warning("Warm-up incomplete - subsequent tests may have cold-start issues")
            # Continue anyway - the actual tests will reveal if there's a real problem

    # Run lifecycle tests (with retry support for any remaining cold-start issues)
    for i in range(num_lifecycle_tests):
        result = validate_sandbox_lifecycle(
            gateway_url,
            test_id=str(i + 1),
            gateway_port=gateway_port,
            max_retries=1,  # Allow one retry per test for transient issues
            retry_delay_sec=5,
        )
        report.add_result(result)

        # Small delay between tests
        if i < num_lifecycle_tests - 1:
            time.sleep(2)

    # Run isolation test
    if include_isolation_test:
        result = validate_sandbox_isolation(gateway_url, gateway_port=gateway_port)
        report.add_result(result)

    logger.info(f"\n{report.summary()}")
    return report


def quick_health_check(
    gateway_url: str,
    gateway_port: int = 443,
    allow_cold_start_retries: bool = True,
) -> bool:
    """Run a single quick health check.

    Args:
        gateway_url: Beta9 gateway URL.
        gateway_port: Gateway gRPC port (443 for Pinggy tunnels, 1993 for direct).
        allow_cold_start_retries: If True, retry on cold-start errors.

    Returns:
        True if health check passes.
    """
    result = validate_sandbox_lifecycle(
        gateway_url,
        test_id="quick",
        gateway_port=gateway_port,
        max_retries=2 if allow_cold_start_retries else 0,
        retry_delay_sec=10,
    )
    return result.passed
