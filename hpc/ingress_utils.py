"""ingress_utils.py — native controller-ingress wiring helpers (capability-URL scheme).

Shared by the RL / datagen launchers and the eval listener when
``--ingress-mode controller`` is selected. Under the default ``pinggy`` mode
NONE of this is reached, so the legacy path stays byte-identical.

The controller path replaces the pinggy tunnel with the native iris
capability-URL scheme that shipped in marin #6857 (``/proxy/t/*``): the co-located
vLLM (or the RecordProxy in the literal combo) is registered with the iris
controller under ``ENDPOINT_ACCESS_LINK``, then a scoped capability token is
minted for it and carried IN THE URL PATH:

    api_base = https://<ingress_host>/proxy/t/<token>/<encoded_endpoint>/v1

Possession of the URL is the credential — there is NO auth header, and the
sandbox-facing ``api_key`` is an unused dummy (installed OpenAI-compatible agents
still require a non-empty key string, so we inject one, but it never rides the
wire). ``<encoded_endpoint>`` is the registered wire name with a leading ``/``
dropped and ``/`` -> ``.`` (the exact encoding of ``rigging.connect.capability_path``
/ ``proxy_path``); our single-segment ``otagent-<slug>`` name encodes to itself.

TOKEN LIFETIME. The controller clamps a minted token to its own
``MAX_ENDPOINT_TOKEN_TTL_SECONDS`` (``DEFAULT`` = 1h). The endpoint
REGISTRATION is separately lease-renewed for the whole run (see
:class:`ControllerEndpointRegistration`); only the token expires. So the api_base
is resolved through :func:`capability_api_base`, which requests that maximum, caches
it worker-side keyed by endpoint name, and re-mints when within
``TOKEN_REFRESH_MARGIN_SECONDS`` of expiry.

INJECTION CADENCE (important). The launchers bake the resolved api_base into the
harbor command once per harbor spawn (``_run_harbor`` -> ``build_harbor_command``),
so it is a PER-HARBOR-SPAWN value, not a per-trial one: a single harbor process
uses one api_base string for its whole lifetime. The worker-side cache therefore
refreshes the token across harbor RE-SPAWNS (resume / campaign refills), not
across trials within one running harbor process. A harbor run that stays up
longer than the token TTL will outlive its token — keep individual harbor runs
within the controller maximum, or re-spawn to re-mint. There is no per-trial base_url resolution hook
in the current OT-Agent->harbor plumbing.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple

from hpc.iris.capability_tokens import controller_max_endpoint_token_ttl_seconds

# The sandbox-facing api_key. The capability token rides in the URL path, so no
# bearer is needed; but installed OpenAI-compatible agents refuse to start
# without SOME non-empty key, so we hand them this inert placeholder. It is
# NOT a secret and never authenticates anything.
DUMMY_API_KEY = "capability-url-no-auth-header"

# Bespoke placeholder var the agent's SANDBOX key is sourced from. Kept SEPARATE
# from OPENAI_API_KEY so the launcher never clobbers the real host OPENAI_API_KEY:
# the nemotron-gym LLM-judge verifiers render `[verifier.env] OPENAI_API_KEY` from
# the WORKER env, so a placeholder there → every judge 401s → uniform reward 0.0
# (2026-07-13, identity-following #29). harbor's opencode agent sources its sandbox
# key from OPENCODE_DUMMY_KEY (see harbor agents/installed/opencode.py).
AGENT_DUMMY_KEY_VAR = "OPENCODE_DUMMY_KEY"
# Legacy vars other OpenAI-compatible agents read directly (qwen/codex/hermes/trae →
# OPENAI_API_KEY, openhands → LLM_API_KEY). Filled ONLY when absent (setdefault) so a
# real host OPENAI_API_KEY survives for the judge while keyless workers still start.
_AGENT_KEY_ENV_VARS = ("OPENAI_API_KEY", "LLM_API_KEY")

# Env var iris sets in-cluster to the externally-visible host a task should
# advertise its services under (``JobInfo.advertise_host``). The controller
# resolves a registered endpoint by connecting to its ``address``, so the
# upstream (raw vLLM, or the co-located RecordProxy) must be reachable at THIS
# host — a loopback ``127.0.0.1`` would only be reachable from inside the task.
# Off-cluster / in tests the var is unset and we fall back to loopback.
ADVERTISE_HOST_ENV = "IRIS_ADVERTISE_HOST"
DEFAULT_ADVERTISE_HOST = "127.0.0.1"

# The raw vLLM HTTP port the RL/datagen servers bind on the task node.
DEFAULT_VLLM_PORT = 8000

# Safety margin at which a cached token is re-minted rather than reused. The
# request lifetime itself is resolved from the controller at runtime.
TOKEN_REFRESH_MARGIN_SECONDS = 2 * 3600  # re-mint when <2h remains


def encode_endpoint_name(name: str) -> str:
    """Encode a wire endpoint name for the ``/proxy`` path (``/`` -> ``.``, leading ``/`` dropped).

    Byte-identical to ``rigging.connect.proxy_path`` / ``capability_path``'s
    ``name.strip('/').replace('/', '.')``. Replicated here (rather than imported)
    so the pure-string helpers stay usable on a launch host whose pinned iris
    predates the capability APIs; the worker runtime (post pin-bump) agrees.
    """
    return name.strip("/").replace("/", ".")


def controller_endpoint_name(job_name: Optional[str]) -> str:
    """Deterministic iris endpoint wire name for a job's vLLM.

    ``otagent-<sanitized-job-name>`` — unique per job, a single DOT-FREE path
    segment, so ``encode_endpoint_name`` is the identity and the registered name,
    the minted token's audience, and the capability path all agree. Anything
    outside ``[A-Za-z0-9_-]`` (including dots and slashes) maps to ``-``.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", (job_name or "job")).strip("-_").lower()
    return f"otagent-{slug or 'job'}"


def build_capability_api_base(ingress_host: str, endpoint_name: str, token: str) -> str:
    """``https://<ingress_host>/proxy/t/<token>/<encoded_endpoint>/v1``.

    The capability URL for an OpenAI server: the scoped token rides in the path
    and no auth header is needed. ``ingress_host`` may omit the scheme.
    """
    host = (ingress_host or "").rstrip("/")
    if not (host.startswith("http://") or host.startswith("https://")):
        host = f"https://{host}"
    return f"{host}/proxy/t/{token}/{encode_endpoint_name(endpoint_name)}/v1"


def inject_ingress_agent_key(env: Optional[dict] = None) -> bool:
    """Publish the inert :data:`DUMMY_API_KEY` for the agents WITHOUT clobbering a
    real ``OPENAI_API_KEY``.

    The capability token is in the URL path, so no real bearer is needed; agents
    just refuse to start without a non-empty key. Sets the bespoke
    :data:`AGENT_DUMMY_KEY_VAR` (``OPENCODE_DUMMY_KEY``) unconditionally, and only
    *fills in* an absent ``OPENAI_API_KEY``/``LLM_API_KEY`` (``setdefault``) — so a
    real host ``OPENAI_API_KEY`` needed by the LLM-judge verifiers is preserved.
    Always returns True (there is no secret to be missing).
    """
    env = os.environ if env is None else env
    # Always publish the bespoke placeholder (harbor's opencode sources it).
    env[AGENT_DUMMY_KEY_VAR] = DUMMY_API_KEY
    # For the legacy agent-facing key vars, only fill an ABSENT value — never
    # overwrite a real host OPENAI_API_KEY (the LLM-judge verifiers need it).
    for var in _AGENT_KEY_ENV_VARS:
        env.setdefault(var, DUMMY_API_KEY)
    return True


# --------------------------------------------------------------------------- #
# Capability-token minting + worker-side cache
# --------------------------------------------------------------------------- #


class CapabilityMinter(Protocol):
    """Mints a scoped capability token for a registered endpoint.

    Returns ``(token, expires_at_epoch_seconds)``. The in-cluster controller-RPC
    adapter and unit-test fakes both satisfy it, so the cache is testable
    in-process without a live controller.
    """

    def mint(self, endpoint_name: str, ttl_hours: float) -> Tuple[str, float]: ...


@dataclass
class _CachedToken:
    token: str
    expires_at: float  # epoch seconds


class CapabilityTokenCache:
    """Worker-side cache of scoped capability tokens, keyed by endpoint name.

    Mints on first use and re-mints when a cached token is within
    :data:`TOKEN_REFRESH_MARGIN_SECONDS` of expiry, so every resolve of the
    api_base hands out a token with ample life left. Thread-safe: harbor spawns
    and lease renewers touch it from different threads.
    """

    def __init__(
        self, minter: CapabilityMinter, *, ttl_hours: float | None = None
    ) -> None:
        self._minter = minter
        # Read the controller-owned maximum lazily instead of maintaining an
        # OT-Agent copy. Tests can still inject an explicit TTL.
        self._ttl_hours = (
            ttl_hours
            if ttl_hours is not None
            else controller_max_endpoint_token_ttl_seconds() / 3600.0
        )
        self._lock = threading.Lock()
        self._cache: Dict[str, _CachedToken] = {}

    def token_for(self, endpoint_name: str, *, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        with self._lock:
            cached = self._cache.get(endpoint_name)
            if (
                cached is not None
                and cached.expires_at - now > TOKEN_REFRESH_MARGIN_SECONDS
            ):
                return cached.token
            token, expires_at = self._minter.mint(endpoint_name, self._ttl_hours)
            if not token:
                raise RuntimeError(
                    f"minting a capability token for {endpoint_name} returned an empty "
                    "token; refusing to build an unreachable api_base."
                )
            self._cache[endpoint_name] = _CachedToken(
                token=token, expires_at=expires_at
            )
            return token


class _ControllerCapabilityMinter:
    """Mints via the in-cluster controller ``MintEndpointToken`` RPC.

    Builds a ``ControllerServiceClientSync`` from the task's
    ``IRIS_CONTROLLER_ADDRESS`` (network-level trust in-cluster, no explicit
    credentials — mirroring the leased ``EndpointClient``). The mint is
    authorized to the endpoint's owning user or an admin; an in-cluster task
    registering its own endpoint is that owner.
    """

    def __init__(self) -> None:
        from iris.cluster.client.job_info import get_job_info
        from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
        from iris.rpc.controller_connect import ControllerServiceClientSync

        info = get_job_info()
        if info is None or not info.controller_address:
            raise RuntimeError(
                "capability-token minting requires an in-cluster iris task "
                "(IRIS_TASK_ID + IRIS_CONTROLLER_ADDRESS); none found."
            )
        self._stub = ControllerServiceClientSync(
            info.controller_address,
            accept_compression=IRIS_RPC_COMPRESSIONS,
            send_compression=None,
        )

    def mint(self, endpoint_name: str, ttl_hours: float) -> Tuple[str, float]:
        from iris.rpc import controller_pb2
        from iris.time_proto import duration_to_proto
        from rigging.timing import Duration

        request = controller_pb2.Controller.MintEndpointTokenRequest(
            endpoint_name=endpoint_name,
            ttl=duration_to_proto(Duration.from_hours(ttl_hours)),
        )
        resp = self._stub.mint_endpoint_token(request)
        return resp.token, resp.expires_at.epoch_ms / 1000.0


# Process-wide cache; the default minter is constructed lazily on first resolve
# (it needs the in-cluster controller address, unavailable at import time).
_TOKEN_CACHE: Optional[CapabilityTokenCache] = None
_TOKEN_CACHE_LOCK = threading.Lock()


def _default_token_cache() -> CapabilityTokenCache:
    global _TOKEN_CACHE
    with _TOKEN_CACHE_LOCK:
        if _TOKEN_CACHE is None:
            _TOKEN_CACHE = CapabilityTokenCache(_ControllerCapabilityMinter())
        return _TOKEN_CACHE


def capability_api_base(
    ingress_host: str,
    endpoint_name: str,
    *,
    cache: Optional[CapabilityTokenCache] = None,
    now: Optional[float] = None,
) -> str:
    """Resolve the current capability api_base for a REGISTERED endpoint.

    Mints (or reuses a cached, still-fresh) scoped token and returns
    ``https://<ingress_host>/proxy/t/<token>/<encoded_endpoint>/v1``. The endpoint
    MUST already be registered with ``ENDPOINT_ACCESS_LINK`` (see
    :func:`register_controller_endpoint`), else the mint has nothing to resolve.
    ``cache`` is injectable for tests; production uses the process-wide worker
    cache backed by the controller RPC.
    """
    token_cache = cache if cache is not None else _default_token_cache()
    token = token_cache.token_for(endpoint_name, now=now)
    return build_capability_api_base(ingress_host, endpoint_name, token)


def build_controller_endpoint_meta(
    ingress_host: str,
    endpoint_name: str,
    *,
    cache: Optional[CapabilityTokenCache] = None,
) -> Dict[str, str]:
    """Endpoint metadata dict for controller mode (capability api_base + dummy key).

    metrics_endpoint is intentionally omitted: the capability route fronts only
    the ``/v1`` inference surface, so ``/metrics`` is not reachable through it.
    """
    return {
        "api_base": capability_api_base(ingress_host, endpoint_name, cache=cache),
        "api_key": DUMMY_API_KEY,
    }


# --------------------------------------------------------------------------- #
# Endpoint registration (shared by the controller path and the +literal combo)
# --------------------------------------------------------------------------- #
#
# The capability api_base only resolves once the vLLM (or, in the literal combo,
# the co-located RecordProxy) is REGISTERED with the iris controller under
# ``<endpoint_name>`` AND with ``ENDPOINT_ACCESS_LINK`` — a PRIVATE (default)
# endpoint rejects a scoped capability token (the controller's proxy authorizer
# returns 403 "endpoint-scoped token cannot access this endpoint"). The helpers
# below register with LINK access and are shared so the plain and the
# record_literal combo register the SAME name (only the address differs).
#
# NAMESPACING: we register through the leased :class:`EndpointClient`
# (``iris.cluster.client.endpoint_client``) directly rather than through
# ``ctx.registry`` — the latter auto-prefixes the name with the job namespace,
# which would break the fixed single-segment ``otagent-<slug>`` name the mint's
# token audience and the capability path both use. ``EndpointClient.register``
# does NOT prefix, so the wire name stays that single segment.
#
# LEASING: iris endpoints are LEASED. ``EndpointClient`` owns the RPC stub AND a
# background ``EndpointLeaseRenewer`` daemon — the lease keeps the controller
# serving the endpoint, and a one-shot register with no renewal expires within
# minutes. So we build a dedicated ``EndpointClient`` (its renewer running),
# register through it with LINK access, and KEEP IT ALIVE for the whole harbor
# run via the returned :class:`ControllerEndpointRegistration`, then ``close()``
# it (stops renewing + unregisters) on run exit. The token expiring is orthogonal
# — the lease renews the registration; the token cache re-mints the token.


class EndpointRegistrar(Protocol):
    """The ``register(name, address, metadata, access) -> endpoint_id`` shape we drive.

    The in-cluster leased-``EndpointClient`` adapter and unit-test fakes both
    satisfy it. An implementation MAY also expose ``close()`` to stop lease
    renewal and unregister; the registration handle calls it on teardown.
    """

    def register(
        self,
        name: str,
        address: str,
        metadata: Optional[Dict[str, str]] = None,
        access: Optional[int] = None,
    ) -> str: ...


@dataclass
class ControllerEndpointRegistration:
    """A live controller endpoint registration whose lease is being renewed.

    Holds the real ``endpoint_id`` and a ``close`` callable that stops the
    background lease renewer and unregisters the endpoint. The caller MUST keep
    this handle alive for the whole harbor run and call :meth:`close` on exit; a
    dropped handle lets the lease lapse and the ``/proxy`` route starts 404-ing.
    """

    endpoint_id: str
    _close: Callable[[], None]

    def close(self) -> None:
        """Stop lease renewal and unregister the endpoint (best-effort, idempotent)."""
        self._close()


class _LeasedEndpointRegistrar:
    """Registers through a dedicated leased iris :class:`EndpointClient`.

    Owns the ``EndpointClient`` (and thus its background ``EndpointLeaseRenewer``
    daemon), so ``register`` returns the real endpoint_id and keeps the lease
    renewed until ``close``. Registers WITHOUT namespace-prefixing and with
    ``ENDPOINT_ACCESS_LINK`` so the capability token minted for the single
    ``otagent-<slug>`` wire name resolves and is accepted by the proxy.
    """

    def __init__(self, client: "EndpointClient", task_attempt: "TaskAttempt") -> None:  # noqa: F821
        self._client = client
        self._task_attempt = task_attempt

    def register(
        self,
        name: str,
        address: str,
        metadata: Optional[Dict[str, str]] = None,
        access: Optional[int] = None,
    ) -> str:
        from iris.cluster.types import EndpointAccess

        access_mode = (
            access if access is not None else EndpointAccess.ENDPOINT_ACCESS_LINK
        )
        return self._client.register(
            name, address, self._task_attempt, metadata or {}, access=access_mode
        )

    def close(self) -> None:
        # Stops the renewer daemon and best-effort unregisters everything still
        # registered, then disconnects the stub.
        self._client.close()


def controller_upstream_address(port: int, *, env: Optional[dict] = None) -> str:
    """``http://<advertise-host>:<port>`` — the address the controller resolves to.

    ``<advertise-host>`` is ``IRIS_ADVERTISE_HOST`` (the task's externally-visible
    host, set by iris in-cluster) falling back to ``127.0.0.1`` off-cluster. The
    controller connects here to reach the registered upstream, so it must NOT be a
    loopback address on a live cluster.
    """
    env = os.environ if env is None else env
    host = env.get(ADVERTISE_HOST_ENV) or DEFAULT_ADVERTISE_HOST
    return f"http://{host}:{port}"


def _default_endpoint_registrar() -> _LeasedEndpointRegistrar:
    """Build the in-cluster leased-``EndpointClient`` registrar.

    Constructs a dedicated ``EndpointClient`` from the task's
    ``IRIS_CONTROLLER_ADDRESS`` (network-level trust in-cluster, no credentials)
    and the task identity from :func:`iris.cluster.client.job_info.get_job_info`.
    Raises loudly (never returns ``None``) when iris is unavailable or we are not
    inside a task — a silent no-op here is exactly what produces a 404-ing run.
    """
    from iris.cluster.client.endpoint_client import EndpointClient
    from iris.cluster.client.job_info import get_job_info
    from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
    from iris.rpc.controller_connect import EndpointServiceClientSync

    info = get_job_info()
    if info is None or not info.controller_address:
        raise RuntimeError(
            "controller endpoint registration requires an in-cluster iris task "
            "(IRIS_TASK_ID + IRIS_CONTROLLER_ADDRESS); none found. Registration "
            "cannot proceed — the /proxy route would 404."
        )
    stub = EndpointServiceClientSync(
        info.controller_address,
        accept_compression=IRIS_RPC_COMPRESSIONS,
        send_compression=None,
    )
    return _LeasedEndpointRegistrar(EndpointClient(stub), info.task_attempt)


def register_controller_endpoint(
    endpoint_name: str,
    address: str,
    *,
    registrar: Optional[EndpointRegistrar] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> ControllerEndpointRegistration:
    """Register ``address`` under ``endpoint_name`` (``ENDPOINT_ACCESS_LINK``).

    Registers through a leased ``EndpointClient`` (its background lease renewer
    keeps the controller serving the endpoint) so the capability route resolves
    to ``address`` for the whole run. LINK access is what lets the minted scoped
    token reach it. ``registrar`` is injectable for unit tests; in production it
    is the in-cluster :func:`_default_endpoint_registrar`.

    Returns a :class:`ControllerEndpointRegistration` handle with the REAL
    ``endpoint_id`` and a ``close()`` that stops renewal + unregisters. The caller
    MUST keep the handle alive for the whole run and ``close()`` it on exit.

    Raises if no registrar is available or the register call yields no id — a
    broken registration must fail loud, not silently 404 the run.
    """
    reg = registrar if registrar is not None else _default_endpoint_registrar()
    endpoint_id = reg.register(endpoint_name, address, metadata or {})
    if not endpoint_id:
        raise RuntimeError(
            f"controller endpoint registration for {endpoint_name} -> {address} "
            "returned no endpoint_id; refusing to proceed (would 404 the run)."
        )
    close = getattr(reg, "close", None)
    return ControllerEndpointRegistration(
        endpoint_id=endpoint_id,
        _close=close if callable(close) else (lambda: None),
    )


def controller_registration_plan(
    job_name: Optional[str],
    *,
    record_literal: bool,
    proxy_port: int,
    vllm_port: int = DEFAULT_VLLM_PORT,
    env: Optional[dict] = None,
) -> Tuple[str, str]:
    """Compute ``(endpoint_name, register_address)`` for controller mode.

    The single decision point the launchers share so the plain and the
    ``record_literal`` combo stay consistent:

      * ``endpoint_name`` — the same ``otagent-<slug>`` either way, so the
        capability api_base (built after register+mint) is stable whether or not
        literal capture is on.
      * ``register_address`` — the co-located RecordProxy's ``proxy_port`` when
        ``record_literal`` is set (controller -> RecordProxy -> vLLM, so literal
        tokens are captured on the served path), otherwise raw vLLM's ``vllm_port``.

    The api_base is NOT returned here: it can only be built AFTER the endpoint is
    registered and a token minted (:func:`capability_api_base`).
    """
    endpoint_name = controller_endpoint_name(job_name)
    port = proxy_port if record_literal else vllm_port
    register_address = controller_upstream_address(port, env=env)
    return endpoint_name, register_address


# --------------------------------------------------------------------------- #
# FEDERATED parent-minting (cross-cluster ingress — Exp2 opencode-RL fix #1)
# --------------------------------------------------------------------------- #
#
# WHY a SEPARATE mint path. The plain :func:`capability_api_base` mints against the
# task's OWN in-cluster controller (``_ControllerCapabilityMinter`` uses
# ``IRIS_CONTROLLER_ADDRESS``). On a CoreWeave peer that controller is the CoreWeave
# controller, whose signing key marin (iris.oa.dev) does NOT trust: federation trust
# is UNIDIRECTIONAL (cw trusts marin, not the reverse), so a cw-minted token 401s at
# iris.oa.dev. And the peer controller's own public host is IP-locked. So a Daytona
# sandbox can only reach the co-located vLLM through iris.oa.dev, which requires:
#
#   1. the job be DELEGATED by marin to the peer (launcher --target-cluster), so
#      marin's ``has_received_job_from_peer`` gate passes and it federation-proxies
#      ``/proxy`` to the peer endpoint;
#   2. the endpoint be registered on the peer (local, exactly as today) and then
#      MIRRORED onto marin by FederationSync (a mirrored row carries a ``peer_id``);
#   3. the capability token be minted at the PARENT (marin) for that mirrored
#      endpoint, so it is signed with marin's key → the ``/proxy/t/<token>/...`` LINK
#      check at iris.oa.dev passes and the request forwards to the peer.
#
# FederationSync mirroring is ASYNC, so between register (2) and mint (3) there is a
# race — :func:`wait_for_endpoint_mirror` bounds it with a poll+timeout and fails
# loud rather than minting a token the parent can't yet resolve. The pure poll/mint
# core takes injectable resolver/minter Protocols (unit-testable with fakes, no live
# controller); the production adapters authenticate to marin via iris's own IAP client
# construction.


# Bounded wait for FederationSync to mirror the peer endpoint onto the parent.
DEFAULT_MIRROR_TIMEOUT_SECONDS = 180.0
DEFAULT_MIRROR_POLL_INTERVAL_SECONDS = 3.0


class ParentEndpointResolver(Protocol):
    """Reports whether ``endpoint_name`` is MIRRORED onto the parent controller yet.

    A mirrored row is one FederationSync copied from a peer — it carries a non-empty
    ``peer_id``. Returns True once such a row exists at the parent. The in-cluster
    parent-controller adapter and unit-test fakes both satisfy it.
    """

    def is_mirrored(self, endpoint_name: str) -> bool: ...


def wait_for_endpoint_mirror(
    endpoint_name: str,
    resolver: ParentEndpointResolver,
    *,
    timeout_s: float = DEFAULT_MIRROR_TIMEOUT_SECONDS,
    interval_s: float = DEFAULT_MIRROR_POLL_INTERVAL_SECONDS,
    sleep: Optional[Callable[[float], None]] = None,
    now: Optional[Callable[[], float]] = None,
) -> None:
    """Block until ``endpoint_name`` is mirrored onto the parent, or raise ``TimeoutError``.

    Polls ``resolver.is_mirrored`` every ``interval_s`` up to ``timeout_s``. This
    bounds the async FederationSync gap between registering the endpoint on the peer
    and minting a token for it at the parent — minting before the mirror appears would
    yield a token the parent cannot resolve (the mint's ``resolve`` returns nothing).
    ``sleep``/``now`` are injectable for deterministic tests.
    """
    _sleep = sleep if sleep is not None else time.sleep
    _now = now if now is not None else time.monotonic
    deadline = _now() + timeout_s
    attempts = 0
    while True:
        attempts += 1
        try:
            if resolver.is_mirrored(endpoint_name):
                return
        except Exception as exc:  # noqa: BLE001 — a transient resolve error should not abort the wait
            last_err: Optional[Exception] = exc
        else:
            last_err = None
        if _now() >= deadline:
            raise TimeoutError(
                f"endpoint {endpoint_name!r} was not mirrored onto the parent controller "
                f"within {timeout_s:.0f}s ({attempts} poll(s)); FederationSync has not "
                "propagated the delegated peer endpoint, so a parent-minted capability "
                "token would not resolve. Is the job actually DELEGATED to the peer "
                "(launcher --target-cluster) and the endpoint registered on the peer?"
                + (f" Last resolve error: {last_err}" if last_err else "")
            )
        _sleep(interval_s)


@dataclass
class _FederatedTokenState:
    mirrored: bool = False


class FederatedCapabilityTokenCache:
    """Parent-minting analog of :class:`CapabilityTokenCache`.

    On the first ``token_for`` for an endpoint it waits (once) for FederationSync to
    mirror the endpoint onto the parent, then mints at the PARENT and caches the token,
    re-minting within :data:`TOKEN_REFRESH_MARGIN_SECONDS` of expiry (the mirror wait
    is NOT repeated on re-mint — a mirrored row does not un-mirror mid-run).
    """

    def __init__(
        self,
        minter: CapabilityMinter,
        resolver: ParentEndpointResolver,
        *,
        ttl_hours: float | None = None,
        mirror_timeout_s: float = DEFAULT_MIRROR_TIMEOUT_SECONDS,
        mirror_interval_s: float = DEFAULT_MIRROR_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._minter = minter
        self._resolver = resolver
        self._ttl_hours = (
            ttl_hours
            if ttl_hours is not None
            else controller_max_endpoint_token_ttl_seconds() / 3600.0
        )
        self._mirror_timeout_s = mirror_timeout_s
        self._mirror_interval_s = mirror_interval_s
        self._lock = threading.Lock()
        self._cache: Dict[str, _CachedToken] = {}
        self._state: Dict[str, _FederatedTokenState] = {}

    def token_for(self, endpoint_name: str, *, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        with self._lock:
            cached = self._cache.get(endpoint_name)
            if (
                cached is not None
                and cached.expires_at - now > TOKEN_REFRESH_MARGIN_SECONDS
            ):
                return cached.token
            state = self._state.setdefault(endpoint_name, _FederatedTokenState())
            if not state.mirrored:
                wait_for_endpoint_mirror(
                    endpoint_name,
                    self._resolver,
                    timeout_s=self._mirror_timeout_s,
                    interval_s=self._mirror_interval_s,
                )
                state.mirrored = True
            token, expires_at = self._minter.mint(endpoint_name, self._ttl_hours)
            if not token:
                raise RuntimeError(
                    f"parent-minting a capability token for {endpoint_name} returned an "
                    "empty token; refusing to build an unreachable api_base."
                )
            self._cache[endpoint_name] = _CachedToken(
                token=token, expires_at=expires_at
            )
            return token


class _ParentControllerClient:
    """Authenticated client to the PARENT (marin) controller for mirror-check + mint.

    Builds an IAP-authenticated ``ControllerServiceClientSync`` against the parent
    cluster config (default: iris.oa.dev / marin.yaml) using iris's own credential
    wiring — the same path ``iris endpoints list`` / ``iris endpoints mint`` use, so
    the mint runs under the caller's identity and the parent owner check passes for a
    ``*@openathena.ai`` submitter. Requires IAP credentials to be available (an
    ``iris login`` refresh token, or an allowlisted service account); raises loudly
    otherwise rather than silently producing an unreachable api_base.

    Satisfies BOTH :class:`ParentEndpointResolver` (``is_mirrored``) and
    :class:`CapabilityMinter` (``mint``).
    """

    def __init__(self, parent_config_path: str) -> None:
        from iris.cluster.config import load_config
        from iris.cli.connect import open_controller_endpoint, rpc_client

        self._config = load_config(parent_config_path)
        # open_controller_endpoint resolves the parent URL + IAP ClientCredentials
        # exactly as the CLI does; rpc_client threads them into the RPC interceptors.
        self._endpoint_cm = open_controller_endpoint(config_file=parent_config_path)
        endpoint = self._endpoint_cm.__enter__()
        self._client = rpc_client(endpoint.url, getattr(endpoint, "credentials", None))

    def is_mirrored(self, endpoint_name: str) -> bool:
        from iris.rpc import controller_pb2

        resp = self._client.list_endpoints(
            controller_pb2.Controller.ListEndpointsRequest(
                prefix=endpoint_name, exact=True
            )
        )
        # A mirrored (federated) row carries a non-empty peer_id; a purely-local
        # parent endpoint of the same name (there should be none) would not.
        return any(getattr(e, "peer_id", "") for e in resp.endpoints)

    def mint(self, endpoint_name: str, ttl_hours: float) -> Tuple[str, float]:
        from iris.rpc import controller_pb2
        from iris.time_proto import duration_to_proto
        from rigging.timing import Duration

        resp = self._client.mint_endpoint_token(
            controller_pb2.Controller.MintEndpointTokenRequest(
                endpoint_name=endpoint_name,
                ttl=duration_to_proto(Duration.from_hours(ttl_hours)),
            )
        )
        return resp.token, resp.expires_at.epoch_ms / 1000.0

    def close(self) -> None:
        try:
            self._endpoint_cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


# Env vars the production wiring reads for the parent (marin) controller.
PARENT_CONTROLLER_CONFIG_ENV = "OTAGENT_PARENT_CONTROLLER_CONFIG"
PARENT_INGRESS_HOST_ENV = "OTAGENT_PARENT_INGRESS_HOST"
DEFAULT_PARENT_INGRESS_HOST = "iris.oa.dev"

_FED_TOKEN_CACHE: Optional[FederatedCapabilityTokenCache] = None
_FED_TOKEN_CACHE_LOCK = threading.Lock()


def _default_federated_token_cache() -> FederatedCapabilityTokenCache:
    """Process-wide federated cache backed by the live parent-controller client.

    Reads the parent config from :data:`PARENT_CONTROLLER_CONFIG_ENV`. Constructed
    lazily (the parent client needs IAP creds unavailable at import time).
    """
    global _FED_TOKEN_CACHE
    with _FED_TOKEN_CACHE_LOCK:
        if _FED_TOKEN_CACHE is None:
            parent_cfg = os.environ.get(PARENT_CONTROLLER_CONFIG_ENV)
            if not parent_cfg:
                raise RuntimeError(
                    "federated parent-minting requires the parent (marin) controller "
                    f"config; set {PARENT_CONTROLLER_CONFIG_ENV} to its cluster YAML "
                    "(the marin.yaml whose dashboard_url is iris.oa.dev)."
                )
            client = _ParentControllerClient(parent_cfg)
            _FED_TOKEN_CACHE = FederatedCapabilityTokenCache(client, client)
        return _FED_TOKEN_CACHE


def federated_capability_api_base(
    endpoint_name: str,
    *,
    ingress_host: Optional[str] = None,
    cache: Optional[FederatedCapabilityTokenCache] = None,
    now: Optional[float] = None,
) -> str:
    """Resolve the capability api_base for a MIRRORED, PARENT-minted federated endpoint.

    Waits (once) for FederationSync to mirror the peer endpoint onto the parent, mints
    a scoped token at the PARENT (marin), and returns
    ``https://<parent_ingress_host>/proxy/t/<token>/<encoded_endpoint>/v1``. The
    endpoint MUST already be registered on the peer (see
    :func:`register_controller_endpoint`, run in-cluster as today) and the job must be
    delegated by marin to the peer (launcher ``--target-cluster``). ``ingress_host``
    defaults to :data:`DEFAULT_PARENT_INGRESS_HOST` / :data:`PARENT_INGRESS_HOST_ENV`;
    it MUST be the marin host (a peer-signed token 401s at iris.oa.dev). ``cache`` is
    injectable for tests; production uses the process-wide parent-authenticated cache.
    """
    host = (
        ingress_host
        or os.environ.get(PARENT_INGRESS_HOST_ENV)
        or DEFAULT_PARENT_INGRESS_HOST
    )
    token_cache = cache if cache is not None else _default_federated_token_cache()
    token = token_cache.token_for(endpoint_name, now=now)
    return build_capability_api_base(host, endpoint_name, token)
