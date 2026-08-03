#!/usr/bin/env python3
"""run_ingress_sidecar_iris.py — on-worker supervisor for the ingress sidecar + pinggy tunnel.

Runs as a persistent, CPU-only Iris job that gives Daytona sandboxes a stable,
auth-gated public door to the on-cluster vLLM. It:

  1. Resolves ``CONTROLLER_PROXY_BASE`` from the Iris-injected
     ``IRIS_CONTROLLER_ADDRESS`` — the controller serves its dashboard +
     ``EndpointProxy`` (``/proxy/<name>/...``) over HTTP on port 10000 at that
     host. Fails loud if the env var is absent (i.e. not running in-cluster).
  2. Starts ``python -m hpc.ingress_sidecar`` (uvicorn) bound to
     ``0.0.0.0:<port>`` — the auth gate that requires a bearer and forwards only
     ``/proxy/<name>/v1`` to the controller EndpointProxy.
  3. Waits for the sidecar's local ``/healthz`` to return 200.
  4. Opens a pinggy tunnel fronting ``127.0.0.1:<port>``, yielding the public
     ``https://<id>.a.pinggy.link`` ingress host.
  5. Prints the public ingress host, then supervises forever: restarts the
     tunnel if it drops; exits non-zero if the sidecar dies, so Iris surfaces
     the failure.

The in-cluster localhost view of the controller proxy is unauthenticated, so
``IRIS_CONTROLLER_AUTH`` is intentionally left unset (the sidecar attaches no
upstream credential). Set it only if a runtime probe shows the controller
returning 401/403 to the sidecar.

Env:
  IRIS_INGRESS_API_KEY     required — sandbox-facing bearer (consumed by the sidecar).
  IRIS_CONTROLLER_ADDRESS  injected by Iris — controller ``host[:port]``; the
                           EndpointProxy is HTTP on port 10000 at that host.
  PINGGY_URL / PINGGY_TOKEN  the pinggy persistent pair (or pass as CLI args).
  SIDECAR_PORT             local sidecar port (default 8443).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Make `hpc` importable regardless of CWD (this file lives in scripts/inference/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hpc.pinggy_utils import PinggyConfig, PinggyTunnel  # noqa: E402

# The controller's dashboard + EndpointProxy are served over HTTP on this port,
# on the same host as the gRPC controller address (IRIS_CONTROLLER_ADDRESS).
CONTROLLER_PROXY_PORT = 10000


def resolve_controller_proxy_base() -> str:
    """Build ``http://<controller-host>:10000`` from the injected controller address.

    Raises:
        SystemExit: if no controller-address env var is present (not in-cluster).
    """
    addr = os.environ.get("IRIS_CONTROLLER_ADDRESS") or os.environ.get("IRIS_CONTROLLER_URL")
    if not addr:
        raise SystemExit(
            "IRIS_CONTROLLER_ADDRESS not set — cannot locate the controller EndpointProxy. "
            "This entrypoint must run as an in-cluster Iris task."
        )
    # addr is host[:rpc_port], optionally with a scheme; keep only the host and
    # re-point at the EndpointProxy HTTP port.
    host = addr.split("://", 1)[-1].split("/", 1)[0]
    host = host.rsplit(":", 1)[0] if ":" in host else host
    return f"http://{host}:{CONTROLLER_PROXY_PORT}"


def start_sidecar(port: int, proxy_base: str) -> subprocess.Popen:
    """Launch `python -m hpc.ingress_sidecar`, streaming its logs to our stdout."""
    env = dict(os.environ)
    env["CONTROLLER_PROXY_BASE"] = proxy_base
    env["SIDECAR_HOST"] = "0.0.0.0"
    env["SIDECAR_PORT"] = str(port)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH", "")]))
    proc = subprocess.Popen(
        [sys.executable, "-m", "hpc.ingress_sidecar"],
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _drain() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write("  [sidecar] " + line)
            sys.stdout.flush()

    threading.Thread(target=_drain, daemon=True).start()
    return proc


def wait_for_healthz(port: int, proc: subprocess.Popen, timeout: int = 60) -> None:
    """Block until the sidecar's local /healthz returns 200, or fail loud."""
    url = f"http://127.0.0.1:{port}/healthz"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"sidecar exited (code {proc.returncode}) before becoming healthy.")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    print(f"[supervisor] sidecar healthy on {url}")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    raise SystemExit(f"sidecar /healthz did not return 200 within {timeout}s.")


def spawn_tunnel(url: str, token: str, port: int, log_path: Path) -> PinggyTunnel:
    """Start a pinggy tunnel fronting 127.0.0.1:<port>; blocks until it binds."""
    cfg = PinggyConfig(
        persistent_url=url,
        token=token,
        local_port=port,
        local_host="127.0.0.1",
        pinggy_host="pro.pinggy.io",
    )
    tunnel = PinggyTunnel(config=cfg, log_path=log_path)
    tunnel.start()
    return tunnel


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pinggy-url", default=os.environ.get("PINGGY_URL"),
                   help="Pinggy persistent url (e.g. abc123.a.pinggy.link). Or set PINGGY_URL.")
    p.add_argument("--pinggy-token", default=os.environ.get("PINGGY_TOKEN"),
                   help="Pinggy auth token for the persistent url. Or set PINGGY_TOKEN.")
    p.add_argument("--port", type=int, default=int(os.environ.get("SIDECAR_PORT", "8443")),
                   help="Local sidecar bind/tunnel port (default 8443).")
    p.add_argument("--healthz-timeout", type=int, default=60,
                   help="Seconds to wait for the sidecar to become healthy.")
    args = p.parse_args()

    if not os.environ.get("IRIS_INGRESS_API_KEY"):
        raise SystemExit("IRIS_INGRESS_API_KEY must be set (pass via `iris job run -e`).")
    if not args.pinggy_url or not args.pinggy_token:
        raise SystemExit("Provide the pinggy pair via --pinggy-url/--pinggy-token or PINGGY_URL/PINGGY_TOKEN.")

    proxy_base = resolve_controller_proxy_base()
    print(f"[supervisor] CONTROLLER_PROXY_BASE = {proxy_base}")

    sidecar = start_sidecar(args.port, proxy_base)
    wait_for_healthz(args.port, sidecar, timeout=args.healthz_timeout)

    log_path = Path("/tmp") / f"pinggy_{args.pinggy_url.split('.')[0]}.log"
    tunnel = spawn_tunnel(args.pinggy_url, args.pinggy_token, args.port, log_path)

    public_host = f"https://{args.pinggy_url}"
    banner = [
        "=" * 72,
        f"  INGRESS_HOST   {public_host}",
        f"  proxy path     {public_host}/proxy/<endpoint_name>/v1",
        "  auth header    Authorization: Bearer $IRIS_INGRESS_API_KEY",
        f"  healthz        {public_host}/healthz",
        f"  upstream       {proxy_base}",
        f"  tunnel log     {log_path}",
        "=" * 72,
    ]
    print("\n".join(banner))
    sys.stdout.flush()

    stopping = threading.Event()

    def _shutdown(signum, _frame) -> None:
        print(f"[supervisor] received signal {signum}; shutting down.")
        stopping.set()
        try:
            tunnel.stop()
        finally:
            if sidecar.poll() is None:
                sidecar.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Supervise: the sidecar dying is fatal (fail the job so Iris shows it); a
    # dropped tunnel is recoverable (pinggy_utils' ssh loop auto-reconnects, but
    # re-spawn as a backstop if the whole process is gone).
    while not stopping.is_set():
        if sidecar.poll() is not None:
            tunnel.stop()
            raise SystemExit(
                f"sidecar process exited (code {sidecar.returncode}); failing job."
            )
        if not tunnel.is_running:
            print("[supervisor] pinggy tunnel process died; re-spawning.")
            try:
                tunnel = spawn_tunnel(args.pinggy_url, args.pinggy_token, args.port, log_path)
            except Exception as exc:  # noqa: BLE001 — retry on next loop, don't crash the supervisor
                print(f"[supervisor] tunnel re-spawn failed: {exc}; retrying.")
        else:
            tunnel._resume_if_stopped()
        time.sleep(10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
