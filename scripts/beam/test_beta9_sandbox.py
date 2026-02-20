#!/usr/bin/env python3
"""
Test script for connecting to an existing Beta9 cluster and running a sandbox.

Usage:
    # Using config file
    python -m scripts.beam.test_beta9_sandbox --config /path/to/beta9_config.ini

    # Using SSH tunnel (run in another terminal first):
    # ssh -L 7000:134.94.199.218:7000 user@jureca-login.fz-juelich.de
    python -m scripts.beam.test_beta9_sandbox --config /path/to/config.ini --host 127.0.0.1

    # Direct connection with overrides
    python -m scripts.beam.test_beta9_sandbox --host 127.0.0.1 --port 7000 --token <TOKEN>

The config file should be an INI file with [default] section containing:
    token = <auth_token>
    gateway_host = <ip_or_hostname>
    gateway_port = <port>
"""

import argparse
import configparser
import socket
import sys
from pathlib import Path


def check_connectivity(host: str, port: int, timeout: float = 5.0) -> tuple[bool, str]:
    """Check if we can connect to the gateway."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            return True, "Connection successful"
        else:
            return False, f"Connection refused or timed out (error code: {result})"
    except socket.gaierror as e:
        return False, f"DNS resolution failed: {e}"
    except socket.timeout:
        return False, "Connection timed out"
    except Exception as e:
        return False, f"Connection error: {e}"


def setup_beta9_config(
    config_path: str | None = None,
    host_override: str | None = None,
    port_override: int | None = None,
    token_override: str | None = None,
) -> tuple[str, int]:
    """Load config from INI file and save to beta9's default location.

    Returns (host, port) tuple for the configured gateway.
    """
    from beta9.config import ConfigContext, save_config

    token = token_override
    gateway_host = host_override
    gateway_port = port_override

    # Load from config file if provided
    if config_path:
        parser = configparser.ConfigParser()
        parser.read(config_path)

        if "default" not in parser.sections():
            raise ValueError(f"Config file must have [default] section: {config_path}")

        section = parser["default"]
        if not token:
            token = section.get("token")
        if not gateway_host:
            gateway_host = section.get("gateway_host")
        if not gateway_port:
            gateway_port = int(section.get("gateway_port", 7000))

    # Validate required fields
    if not token:
        raise ValueError("Token is required (--token or config file)")
    if not gateway_host:
        raise ValueError("Gateway host is required (--host or config file)")
    if not gateway_port:
        gateway_port = 7000

    ctx = ConfigContext(
        token=token,
        gateway_host=gateway_host,
        gateway_port=gateway_port,
    )

    # Create .beta9 directory if needed
    config_dir = Path.home() / ".beta9"
    config_dir.mkdir(exist_ok=True)

    # Save config
    save_config({"default": ctx})
    print(f"Configured beta9 to connect to {gateway_host}:{gateway_port}")

    return gateway_host, gateway_port


def test_sandbox() -> bool:
    """Create a sandbox, run a simple command, and clean up."""
    from beta9 import Image, Sandbox

    print("\n=== Creating Sandbox ===")
    print("Specs: 1 CPU, 2048MB memory, 2048MB storage")

    sandbox = Sandbox(
        cpu=1,
        memory="2Gi",
        image=Image(python_version="python3.11"),
        keep_warm_seconds=300,  # 5 minutes
    )

    instance = None
    try:
        print("Creating sandbox instance...")
        instance = sandbox.create()
        print("Sandbox created successfully!")

        # Run a simple test command
        print("\n=== Running Test Command ===")
        response = instance.process.run_code(
            """
import os
import platform

print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"CPU count: {os.cpu_count()}")
print("Hello from Beta9 sandbox!")
"""
        )

        print(f"Exit code: {response.exit_code}")
        print(f"Output:\n{response.result}")

        if response.exit_code != 0:
            print(f"Stderr: {response.stderr}")
            return False

        # Test file system operations using shell commands
        # (fs.upload_file/download_file are for local<->sandbox transfers,
        # not for creating files inside the sandbox)
        print("\n=== Testing File System ===")
        test_content = "Hello from Beta9 test!"

        # Write file using shell
        write_response = instance.process.run_code(f"""
with open('/tmp/test.txt', 'w') as f:
    f.write('{test_content}')
print('File written successfully')
""")
        if write_response.exit_code != 0:
            print(f"Failed to write file: {write_response.stderr}")
            return False

        # Read file using shell
        read_response = instance.process.run_code("""
with open('/tmp/test.txt', 'r') as f:
    content = f.read()
print(content, end='')
""")
        if read_response.exit_code != 0:
            print(f"Failed to read file: {read_response.stderr}")
            return False

        read_content = read_response.result.strip()
        print(f"Wrote and read file: '{read_content}'")

        if read_content != test_content:
            print(f"File content mismatch! Expected '{test_content}', got '{read_content}'")
            return False

        print("\n=== Test Passed! ===")
        return True

    except Exception as e:
        print(f"\n=== Test Failed ===")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        print("\n=== Cleaning Up ===")
        if instance is not None:
            try:
                instance.terminate()
                print("Sandbox terminated.")
            except Exception as e:
                print(f"Warning: Failed to terminate sandbox: {e}")
        else:
            print("No sandbox instance to clean up.")


def main():
    parser = argparse.ArgumentParser(
        description="Test Beta9 cluster connectivity by creating a sandbox"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to beta9 config INI file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override gateway host (useful for SSH tunnels, e.g., 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override gateway port (default: 7000)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Override auth token",
    )
    parser.add_argument(
        "--skip-connectivity-check",
        action="store_true",
        help="Skip the initial connectivity check",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.config and not (args.host and args.token):
        # Try default config path
        default_config = Path("/Users/benjaminfeuer/Documents/beta9_config.ini")
        if default_config.exists():
            args.config = str(default_config)
        else:
            print("Error: Must provide --config or both --host and --token")
            sys.exit(1)

    if args.config and not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Setup config
    if args.config:
        print(f"Loading config from: {args.config}")
    host, port = setup_beta9_config(
        config_path=args.config,
        host_override=args.host,
        port_override=args.port,
        token_override=args.token,
    )

    # Check connectivity
    if not args.skip_connectivity_check:
        print(f"\n=== Checking Connectivity ===")
        print(f"Testing connection to {host}:{port}...")
        connected, message = check_connectivity(host, port)

        if not connected:
            print(f"FAILED: {message}")
            print("\n=== Troubleshooting ===")
            print("The gateway is not reachable. Possible causes:")
            print("  1. Firewall blocking the port")
            print("  2. Cluster only accessible from internal network")
            print("  3. Gateway not running")
            print("\nSolutions:")
            print("  - Use SSH tunnel:")
            print(f"      ssh -L {port}:{host}:{port} user@login-node")
            print(f"      python -m scripts.beam.test_beta9_sandbox --config {args.config} --host 127.0.0.1")
            print("  - Use VPN to access internal network")
            print("  - Ask cluster admin to expose the port")
            sys.exit(1)

        print(f"OK: {message}")

    # Run test
    success = test_sandbox()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
