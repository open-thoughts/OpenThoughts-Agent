"""
Cloud provider utilities for SkyPilot-based cloud launches.

This module provides:
- Provider configuration and metadata
- Cloud object resolution
- Provider-specific setup instructions
- Capability checks

Supported providers:
- Major clouds: gcp, aws, azure
- GPU clouds: lambda, vast, runpod, cudo, paperspace, fluidstack
- Infrastructure: kubernetes

For setup instructions, run:
    python -m hpc.cloud_providers --check
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import sky


@dataclass
class ProviderConfig:
    """Configuration and metadata for a cloud provider."""

    name: str
    display_name: str
    cloud_class: str  # Name of sky.clouds.X class
    supports_docker_runtime: bool = True
    supports_spot: bool = True
    supports_regions: bool = True
    default_accelerator: Optional[str] = None
    setup_instructions: List[str] = field(default_factory=list)
    credential_paths: List[str] = field(default_factory=list)
    pip_extras: List[str] = field(default_factory=list)


# Provider configurations
PROVIDERS: Dict[str, ProviderConfig] = {
    # Major cloud providers
    "gcp": ProviderConfig(
        name="gcp",
        display_name="Google Cloud Platform",
        cloud_class="GCP",
        default_accelerator="A100:1",
        setup_instructions=[
            "pip install google-api-python-client",
            "Install gcloud SDK: brew install --cask google-cloud-sdk",
            "gcloud init",
            "gcloud auth application-default login",
        ],
        pip_extras=["google-api-python-client"],
    ),
    "aws": ProviderConfig(
        name="aws",
        display_name="Amazon Web Services",
        cloud_class="AWS",
        default_accelerator="A100:1",
        setup_instructions=[
            "pip install boto3",
            "aws configure",
            "aws configure list  # Verify identity is set",
        ],
        credential_paths=["~/.aws/credentials"],
        pip_extras=["boto3"],
    ),
    "azure": ProviderConfig(
        name="azure",
        display_name="Microsoft Azure",
        cloud_class="Azure",
        default_accelerator="A100:1",
        setup_instructions=[
            "pip install azure-cli",
            "az login",
            "az account set -s <subscription_id>",
        ],
        credential_paths=["~/.azure/msal_token_cache.json"],
        pip_extras=["azure-cli", "azure-identity"],
    ),
    # GPU cloud providers
    "lambda": ProviderConfig(
        name="lambda",
        display_name="Lambda Cloud",
        cloud_class="Lambda",
        supports_spot=False,  # Lambda doesn't have spot instances
        supports_regions=False,  # Limited region support
        default_accelerator="A100:1",
        setup_instructions=[
            "Get API key from: https://cloud.lambdalabs.com/api-keys",
            "mkdir -p ~/.lambda_cloud",
            'echo "api_key = <YOUR_API_KEY>" > ~/.lambda_cloud/lambda_keys',
        ],
        credential_paths=["~/.lambda_cloud/lambda_keys"],
    ),
    "vast": ProviderConfig(
        name="vast",
        display_name="Vast.ai",
        cloud_class="Vast",
        supports_spot=True,  # Vast has interruptible instances
        supports_regions=False,
        default_accelerator="RTX4090:1",
        setup_instructions=[
            "pip install 'vastai-sdk>=0.1.12'",
            "Get API key from: https://cloud.vast.ai/account/",
            "echo '<YOUR_API_KEY>' > ~/.vast_api_key",
        ],
        credential_paths=["~/.vast_api_key"],
        pip_extras=["vastai-sdk>=0.1.12"],
    ),
    "runpod": ProviderConfig(
        name="runpod",
        display_name="RunPod",
        cloud_class="RunPod",
        supports_docker_runtime=False,  # RunPod doesn't support docker as runtime
        supports_spot=True,
        supports_regions=False,
        default_accelerator="A100:1",
        setup_instructions=[
            "pip install 'skypilot[runpod]'",
            "Get API key from: https://www.runpod.io/console/user/settings",
            "mkdir -p ~/.runpod",
            "echo '<YOUR_API_KEY>' > ~/.runpod/api_key",
        ],
        credential_paths=["~/.runpod/api_key"],
    ),
    "cudo": ProviderConfig(
        name="cudo",
        display_name="Cudo Compute",
        cloud_class="Cudo",
        supports_spot=False,
        supports_regions=True,
        default_accelerator="A100:1",
        setup_instructions=[
            "pip install cudo-compute",
            "Get API key from: https://www.cudocompute.com/",
            "cudo auth login",
        ],
        pip_extras=["cudo-compute"],
    ),
    "paperspace": ProviderConfig(
        name="paperspace",
        display_name="Paperspace",
        cloud_class="Paperspace",
        supports_spot=False,
        supports_regions=True,
        default_accelerator="A100:1",
        setup_instructions=[
            "Get API key from: https://console.paperspace.com/",
            "mkdir -p ~/.paperspace",
            'echo \'{"apiKey": "<YOUR_API_KEY>"}\' > ~/.paperspace/config.json',
        ],
        credential_paths=["~/.paperspace/config.json"],
    ),
    "fluidstack": ProviderConfig(
        name="fluidstack",
        display_name="FluidStack",
        cloud_class="Fluidstack",
        supports_spot=False,
        supports_regions=False,
        default_accelerator="A100:1",
        setup_instructions=[
            "Get API key from: https://dashboard.fluidstack.io",
            "mkdir -p ~/.fluidstack",
            "echo '<YOUR_API_KEY>' > ~/.fluidstack/api_key",
        ],
        credential_paths=["~/.fluidstack/api_key"],
    ),
    # Infrastructure providers
    "kubernetes": ProviderConfig(
        name="kubernetes",
        display_name="Kubernetes",
        cloud_class="Kubernetes",
        supports_spot=False,
        supports_regions=False,  # Uses contexts instead
        default_accelerator="A100:1",
        setup_instructions=[
            "Ensure kubectl is configured with a valid kubeconfig",
            "kubeconfig should be at ~/.kube/config",
            "For GKE: gcloud container clusters get-credentials <cluster>",
            "For EKS: aws eks update-kubeconfig --name <cluster>",
            "On macOS, also install: brew install socat netcat",
        ],
        credential_paths=["~/.kube/config"],
    ),
}

# Aliases for common names
PROVIDER_ALIASES: Dict[str, str] = {
    "google": "gcp",
    "amazon": "aws",
    "lambdalabs": "lambda",
    "lambda_cloud": "lambda",
    "vastai": "vast",
    "vast.ai": "vast",
    "k8s": "kubernetes",
    "kube": "kubernetes",
}


def resolve_provider_name(name: str) -> str:
    """Resolve provider name, handling aliases."""
    name_lower = name.lower()
    if name_lower in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[name_lower]
    if name_lower in PROVIDERS:
        return name_lower
    raise ValueError(
        f"Unknown provider '{name}'. "
        f"Supported: {', '.join(sorted(PROVIDERS.keys()))}"
    )


def get_provider_config(name: str) -> ProviderConfig:
    """Get configuration for a provider."""
    resolved = resolve_provider_name(name)
    return PROVIDERS[resolved]


def resolve_cloud(name: str) -> "sky.Cloud":
    """Resolve provider name to a SkyPilot cloud object."""
    import sky

    config = get_provider_config(name)
    cloud_cls = getattr(sky.clouds, config.cloud_class, None)
    if cloud_cls is None:
        raise ValueError(
            f"SkyPilot does not have cloud class '{config.cloud_class}' "
            f"for provider '{name}'. You may need to upgrade SkyPilot."
        )
    return cloud_cls()


def check_provider_credentials(name: str) -> tuple[bool, str]:
    """Check if credentials are configured for a provider.

    Returns:
        (is_configured, message)
    """
    config = get_provider_config(name)

    # Check credential files exist
    missing_files = []
    for cred_path in config.credential_paths:
        expanded = Path(cred_path).expanduser()
        if not expanded.exists():
            missing_files.append(cred_path)

    if missing_files:
        return False, f"Missing credential files: {', '.join(missing_files)}"

    return True, "Credentials appear to be configured"


def get_setup_instructions(name: str) -> str:
    """Get setup instructions for a provider."""
    config = get_provider_config(name)
    lines = [f"Setup instructions for {config.display_name}:", ""]
    for i, instruction in enumerate(config.setup_instructions, 1):
        lines.append(f"  {i}. {instruction}")
    return "\n".join(lines)


def list_providers(verbose: bool = False) -> str:
    """List all supported providers."""
    lines = ["Supported cloud providers:", ""]

    # Group by category
    major_clouds = ["gcp", "aws", "azure"]
    gpu_clouds = ["lambda", "vast", "runpod", "cudo", "paperspace", "fluidstack"]
    infra = ["kubernetes"]

    def format_provider(name: str) -> str:
        config = PROVIDERS[name]
        flags = []
        if not config.supports_docker_runtime:
            flags.append("no-docker")
        if not config.supports_spot:
            flags.append("no-spot")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"  {name:12} - {config.display_name}{flag_str}"

    lines.append("Major clouds:")
    for p in major_clouds:
        lines.append(format_provider(p))

    lines.append("")
    lines.append("GPU clouds:")
    for p in gpu_clouds:
        lines.append(format_provider(p))

    lines.append("")
    lines.append("Infrastructure:")
    for p in infra:
        lines.append(format_provider(p))

    if verbose:
        lines.append("")
        lines.append("Aliases:")
        for alias, target in sorted(PROVIDER_ALIASES.items()):
            lines.append(f"  {alias} -> {target}")

    return "\n".join(lines)


def check_all_providers() -> str:
    """Check credentials for all providers and return status report."""
    lines = ["Provider credential status:", ""]

    for name, config in PROVIDERS.items():
        is_ok, msg = check_provider_credentials(name)
        status = "OK" if is_ok else "MISSING"
        lines.append(f"  [{status:7}] {config.display_name} ({name})")
        if not is_ok:
            lines.append(f"           {msg}")

    return "\n".join(lines)


def get_all_provider_names() -> List[str]:
    """Get list of all provider names (for argparse choices)."""
    return sorted(PROVIDERS.keys())


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cloud provider utilities")
    parser.add_argument("--list", "-l", action="store_true", help="List all providers")
    parser.add_argument("--check", "-c", action="store_true", help="Check all credentials")
    parser.add_argument("--setup", "-s", metavar="PROVIDER", help="Show setup instructions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.setup:
        print(get_setup_instructions(args.setup))
    elif args.check:
        print(check_all_providers())
    elif args.list or not any([args.setup, args.check]):
        print(list_providers(verbose=args.verbose))
