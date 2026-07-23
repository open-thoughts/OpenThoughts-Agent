"""Controller capability-token policy shared by Iris datagen and eval launchers.

The controller owns the maximum lifetime for endpoint-scoped proxy tokens.  Do
not duplicate its value here: an image/controller update may safely change the
limit without an OT-Agent source edit.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import yaml

from hpc.local_paths import PATHS, ensure as ensure_local_paths


_CONTROLLER_AUTH_MODULE = "iris.cluster.controller.auth"
_MAX_TTL_ATTRIBUTE = "MAX_ENDPOINT_TOKEN_TTL_SECONDS"
_POLICY_SCHEMA_VERSION = 1


def is_token_required_agent(agent: str | None) -> bool:
    """Whether an agent must call a capability URL from outside the Iris VPC."""
    return bool(agent) and agent.strip().lower() != "terminus-2"


def infer_harbor_agent(
    agent: str | None, harbor_config: str | Path | None
) -> str | None:
    """Use the configured first Harbor agent when the CLI omitted ``--agent``."""
    if agent:
        return agent
    if not harbor_config:
        return None
    try:
        loaded = yaml.safe_load(Path(harbor_config).read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Could not read Harbor config {harbor_config!s}: {exc}"
        ) from exc
    agents = loaded.get("agents") if isinstance(loaded, dict) else None
    first = agents[0] if isinstance(agents, list) and agents else None
    name = first.get("name") if isinstance(first, dict) else None
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "Token-duration enforcement needs --agent or Harbor config agents[0].name."
        )
    return name


def controller_max_endpoint_token_ttl_seconds(
    *, module_loader: Callable[[str], object] = importlib.import_module
) -> int:
    """Read the authoritative maximum directly from the installed Marin codebase."""
    module = module_loader(_CONTROLLER_AUTH_MODULE)
    value = getattr(module, _MAX_TTL_ATTRIBUTE, None)
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"{_CONTROLLER_AUTH_MODULE}.{_MAX_TTL_ATTRIBUTE} must be a positive integer; "
            f"got {value!r}."
        )
    return value


@dataclass(frozen=True)
class CapabilityTokenDurationPolicy:
    """Secret-free, persisted launch policy for a single Iris workload."""

    schema_version: int
    agent: str | None
    token_required: bool
    controller_max_ttl_seconds: int | None
    requested_timeout_seconds: int
    effective_timeout_seconds: int
    requested_token_ttl_seconds: int | None
    effective_token_ttl_seconds: int | None
    controller_auth_module: str | None

    def to_dict(self) -> dict[str, int | str | bool | None]:
        return asdict(self)


def resolve_token_duration_policy(
    *,
    agent: str | None,
    timeout_seconds: int,
    requested_token_ttl_seconds: int | None = None,
    max_ttl_resolver: Callable[[], int] = controller_max_endpoint_token_ttl_seconds,
) -> CapabilityTokenDurationPolicy:
    """Bound a token-required launch to its actual minted-token lifetime.

    ``timeout_seconds=0`` is Iris's historical "unlimited" spelling.  It is
    safe only for an in-process harness; token-required jobs receive the
    controller maximum by default.  A caller requesting a shorter token gets a
    correspondingly shorter job bound.
    """
    if timeout_seconds < 0:
        raise ValueError("Iris timeout must be zero or a positive number of seconds.")
    required = is_token_required_agent(agent)
    if not required:
        return CapabilityTokenDurationPolicy(
            schema_version=_POLICY_SCHEMA_VERSION,
            agent=agent,
            token_required=False,
            controller_max_ttl_seconds=None,
            requested_timeout_seconds=timeout_seconds,
            effective_timeout_seconds=timeout_seconds,
            requested_token_ttl_seconds=None,
            effective_token_ttl_seconds=None,
            controller_auth_module=None,
        )

    maximum = max_ttl_resolver()
    if requested_token_ttl_seconds is not None and requested_token_ttl_seconds <= 0:
        raise ValueError("Requested capability-token TTL must be positive.")
    if (
        requested_token_ttl_seconds is not None
        and requested_token_ttl_seconds > maximum
    ):
        raise ValueError(
            "Requested capability-token TTL "
            f"({requested_token_ttl_seconds}s) exceeds Marin controller maximum ({maximum}s)."
        )
    effective_token_ttl = requested_token_ttl_seconds or maximum
    effective_timeout = timeout_seconds or effective_token_ttl
    if effective_timeout > effective_token_ttl:
        raise ValueError(
            "Token-required Iris jobs cannot outlive their minted endpoint token: "
            f"requested timeout={effective_timeout}s, token TTL={effective_token_ttl}s."
        )
    return CapabilityTokenDurationPolicy(
        schema_version=_POLICY_SCHEMA_VERSION,
        agent=agent,
        token_required=True,
        controller_max_ttl_seconds=maximum,
        requested_timeout_seconds=timeout_seconds,
        effective_timeout_seconds=effective_timeout,
        requested_token_ttl_seconds=requested_token_ttl_seconds,
        effective_token_ttl_seconds=effective_token_ttl,
        controller_auth_module=_CONTROLLER_AUTH_MODULE,
    )


def persist_token_duration_policy(
    *, job_name: str, policy: CapabilityTokenDurationPolicy, root: Path | None = None
) -> Path:
    """Write the launch policy in the managed local control-plane state tree."""
    state_root = root or (PATHS.state / "iris_capability_token_policies")
    if root is None:
        ensure_local_paths(PATHS.home, PATHS.state, state_root)
    else:
        state_root.mkdir(parents=True, exist_ok=True)
    destination = state_root / f"{job_name}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(policy.to_dict(), sort_keys=True, indent=2) + "\n")
    os.replace(temporary, destination)
    return destination
