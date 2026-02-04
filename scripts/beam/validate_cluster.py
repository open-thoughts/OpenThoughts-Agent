"""Sandbox health check validation for Beta9 cluster."""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


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


def configure_beta9_endpoint(gateway_url: str, gateway_port: int = 1993, token: str = "local-test-token"):
    """Configure beta9 SDK to use custom gateway endpoint.

    This properly saves the config to ~/.beta9/ so the SDK uses the correct
    gateway host and port for gRPC connections.

    Args:
        gateway_url: Full URL to Beta9 gateway (e.g., https://mybeam.a.pinggy.link).
        gateway_port: Gateway gRPC port (default 1993, or 443 for Pinggy tunnels).
        token: Auth token for Beta9 gateway.
    """
    from pathlib import Path

    try:
        from beta9.config import ConfigContext, save_config
    except ImportError:
        logger.warning("beta9 SDK not installed, skipping config save")
        return

    # Extract hostname from URL
    gateway_host = gateway_url.replace("https://", "").replace("http://", "").rstrip("/")

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


def validate_sandbox_lifecycle(
    gateway_url: str,
    test_id: str = None,
    gateway_port: int = 443,
) -> ValidationResult:
    """Test sandbox create -> exec -> terminate lifecycle.

    Args:
        gateway_url: Beta9 gateway URL.
        test_id: Optional test identifier.
        gateway_port: Gateway gRPC port (443 for Pinggy tunnels, 1993 for direct).

    Returns:
        ValidationResult with test outcome.
    """
    test_name = f"sandbox_lifecycle_{test_id or uuid4().hex[:6]}"
    start_time = time.time()
    instance = None
    stdout = None

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
        duration = time.time() - start_time
        return ValidationResult(
            test_name=test_name,
            passed=False,
            duration_sec=duration,
            error=str(e),
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


def validate_sandbox_isolation(gateway_url: str, gateway_port: int = 443) -> ValidationResult:
    """Test that sandboxes are properly isolated.

    Creates two sandboxes and verifies they cannot see each other's files.

    Args:
        gateway_url: Beta9 gateway URL.
        gateway_port: Gateway gRPC port (443 for Pinggy tunnels, 1993 for direct).

    Returns:
        ValidationResult with test outcome.
    """
    test_name = "sandbox_isolation"
    start_time = time.time()
    instance1 = None
    instance2 = None

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
        duration = time.time() - start_time
        return ValidationResult(
            test_name=test_name,
            passed=False,
            duration_sec=duration,
            error=str(e),
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


def run_validation_suite(
    gateway_url: str,
    num_lifecycle_tests: int = 3,
    include_isolation_test: bool = True,
    gateway_port: int = 443,
) -> ValidationReport:
    """Run full validation suite.

    Args:
        gateway_url: Beta9 gateway URL.
        num_lifecycle_tests: Number of sandbox lifecycle tests to run.
        include_isolation_test: Whether to include isolation test.
        gateway_port: Gateway gRPC port (443 for Pinggy tunnels, 1993 for direct).

    Returns:
        ValidationReport with all test results.
    """
    report = ValidationReport()

    logger.info(f"Running validation suite against: {gateway_url}")
    logger.info(f"  Lifecycle tests: {num_lifecycle_tests}")
    logger.info(f"  Isolation test: {include_isolation_test}")
    logger.info(f"  Gateway port: {gateway_port}")

    # Run lifecycle tests
    for i in range(num_lifecycle_tests):
        result = validate_sandbox_lifecycle(gateway_url, test_id=str(i + 1), gateway_port=gateway_port)
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


def quick_health_check(gateway_url: str) -> bool:
    """Run a single quick health check.

    Args:
        gateway_url: Beta9 gateway URL.

    Returns:
        True if health check passes.
    """
    result = validate_sandbox_lifecycle(gateway_url, test_id="quick")
    return result.passed
