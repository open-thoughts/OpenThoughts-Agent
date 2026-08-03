#!/usr/bin/env python3
"""deploy_ingress_sidecar.py — lifecycle CLI for the ingress-sidecar Iris job.

Runs on the Mac / launch host. Manages the persistent, CPU-only,
non-preemptible ``ingress-sidecar`` Iris job (the auth-gating
pinggy->controller-EndpointProxy front door for Daytona sandboxes).

Subcommands:
  deploy   Idempotent stand-up. If a healthy ingress job already exists, print
           its host and exit 0. Otherwise pick a FREE pinggy pair from the bank,
           launch the CPU-only Iris job, wait for it to serve, parse the
           ingress_host from the job logs, run the G2 verification (401 without
           key / not-401 with key / healthz 200), and print
           ``INGRESS_HOST=... JOB_ID=...``.
  ensure   Cron-maintenance path. Check the Iris job state AND the public
           /healthz; if the job isn't running or healthz fails, redeploy and
           re-emit the new host. Exit 0 iff already healthy (no action);
           non-zero if it had to act or could not recover.
  status   Print job_id + state + ingress_host + healthz. No side effects.

Secrets: the sandbox-facing bearer is read from ``IRIS_INGRESS_API_KEY`` in the
environment (``source "$DC_AGENT_SECRET_ENV"`` first; see .agents/secret.md). It is
never printed.

Examples:
  source "${DC_AGENT_SECRET_ENV:?see .agents/secret.md}"
  python scripts/inference/deploy_ingress_sidecar.py deploy
  python scripts/inference/deploy_ingress_sidecar.py ensure   # cron
  python scripts/inference/deploy_ingress_sidecar.py status
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# serve_public.py (same dir) already parses the pinggy bank; reuse it.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from serve_public import parse_bank  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]

IRIS_BIN = os.environ.get("IRIS_BIN", "/Users/benjaminfeuer/Documents/marin/.venv/bin/iris")
# Cluster the sidecar runs on. It fronts THAT cluster's controller EndpointProxy, so it must
# be the SAME cluster the served endpoint is registered on (a cross-cluster/federated proxy
# hop is exactly what marin#7448 broke). Override for a CoreWeave-GPU endpoint (e.g. the grug
# fork serve on cw-us-east-02a) via IRIS_INGRESS_CLUSTER.
CLUSTER = os.environ.get("IRIS_INGRESS_CLUSTER", "marin")
JOB_NAME = "ingress-sidecar"
ENTRYPOINT = "scripts/inference/run_ingress_sidecar_iris.py"
# Region constraint for the sidecar job. Empty ⇒ omit --region (CoreWeave clusters are
# single-region and reject the GCP-TPU region names). Override via IRIS_INGRESS_REGION.
DEFAULT_REGION = os.environ.get("IRIS_INGRESS_REGION", "us-east5")
DEFAULT_BANK = os.environ.get(
    "PINGGY_BANK", str(Path.home() / "Documents" / "notes" / "ot-agent" / "pinggy_bank.md")
)
SIDECAR_PORT = 8443


def log(msg: str) -> None:
    """Diagnostic to stderr; machine-greppable results go to stdout."""
    print(msg, file=sys.stderr, flush=True)


def require_key() -> str:
    key = os.environ.get("IRIS_INGRESS_API_KEY")
    if not key:
        raise SystemExit(
            "IRIS_INGRESS_API_KEY not in env — "
            "`source \"$DC_AGENT_SECRET_ENV\"` first (see .agents/secret.md)."
        )
    return key


def run_iris(*args: str, cwd: str | None = None, timeout: int = 300, check: bool = True) -> str:
    """Invoke the iris CLI; return stdout (iris logs go to stderr)."""
    result = subprocess.run(
        [IRIS_BIN, "--cluster", CLUSTER, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise SystemExit(f"iris {' '.join(args)} failed (code {result.returncode}):\n{result.stderr[-1200:]}")
    return result.stdout


# --------------------------------------------------------------------------- #
# Public-URL probing (via the real pinggy edge, bypassing local DNS poisoning).
# --------------------------------------------------------------------------- #
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_RESOLVERS = ("1.1.1.1", "8.8.8.8", "9.9.9.9")


def _bare_host(host: str) -> str:
    """Strip any scheme/path so we have just the DNS name (banner gives a full URL)."""
    return host.split("://", 1)[-1].split("/", 1)[0]


def _edge_ip(host: str) -> str:
    """Resolve the pinggy edge IPv4 via public resolvers (this Mac's ISP poisons
    ``*.a.pinggy.link``).

    Not cached: pinggy migrates a persistent subdomain's edge IP over time, so a
    cached IP goes stale after a migration. dig is fast, so re-resolve fresh
    each call and fall through several resolvers for robustness.
    """
    host = _bare_host(host)
    for resolver in _RESOLVERS:
        try:
            lines = subprocess.run(
                ["dig", f"@{resolver}", "+short", "+tries=2", "+time=3", host],
                capture_output=True, text=True, timeout=12,
            ).stdout.splitlines()
        except Exception:
            continue
        ips = [ln.strip() for ln in lines if _IPV4_RE.match(ln.strip())]
        if ips:
            return ips[-1]
    return ""


def curl_code(host: str, path: str, bearer: str | None = None, timeout: int = 25, retries: int = 3) -> int:
    """HTTP status of ``https://<host><path>`` via the pinggy edge; 0 if no response.

    Always forces a freshly-resolved edge IP with ``--resolve`` (system DNS is
    poisoned here). Re-resolves on every retry so a mid-check edge migration is
    tracked rather than read as 'down'.
    """
    host = _bare_host(host)
    code = 0
    for _ in range(retries):
        edge = _edge_ip(host)
        cmd = ["curl", "-sk", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", str(timeout)]
        if edge:
            cmd += ["--resolve", f"{host}:443:{edge}"]
        if bearer:
            cmd += ["-H", f"Authorization: Bearer {bearer}"]
        cmd.append(f"https://{host}{path}")
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10).stdout.strip()
            code = int(out or "0")
        except Exception:
            code = 0
        if code != 0:
            return code
        time.sleep(2)
    return code


def is_public_healthy(host: str, tolerance: int = 45) -> bool:
    """True if the public /healthz returns 200 within ``tolerance`` seconds.

    Tolerant window so a brief pinggy edge migration (a few seconds of 000)
    doesn't read as down.
    """
    deadline = time.time() + tolerance
    while True:
        if curl_code(host, "/healthz", timeout=15) == 200:
            return True
        if time.time() >= deadline:
            return False
        time.sleep(5)


def healthz_ok(host: str) -> bool:
    return is_public_healthy(host)


def g2_verify(host: str, api_key: str) -> tuple[bool, dict]:
    """Run the three G2 checks: no-bearer 401, with-bearer not-401, healthz 200."""
    no_bearer = curl_code(host, "/proxy/nope/v1/models")
    with_bearer = curl_code(host, "/proxy/nope/v1/models", bearer=api_key)
    healthz = curl_code(host, "/healthz")
    ok = no_bearer == 401 and with_bearer not in (0, 401) and healthz == 200
    return ok, {"no_bearer": no_bearer, "with_bearer": with_bearer, "healthz": healthz}


# --------------------------------------------------------------------------- #
# Pinggy pair selection.
# --------------------------------------------------------------------------- #
def pick_free_pair(bank: str, exclude: set[str]) -> tuple[str, str]:
    """Return a FREE (url, token) pair from the bank.

    A persistent pinggy subdomain with no bound tunnel does not answer HTTP
    (curl_code -> 0) when reached via its real edge IP; a bound tunnel returns a
    real status. So code 0 == FREE.
    """
    for url, token in parse_bank(bank):
        if url in exclude:
            continue
        if curl_code(url, "/healthz", timeout=12) == 0:
            return url, token
    raise SystemExit(f"No free pinggy pair found in {bank} (all appear in use).")


# --------------------------------------------------------------------------- #
# Iris job discovery / lifecycle.
# --------------------------------------------------------------------------- #
def find_running_job() -> dict | None:
    """The running ingress-sidecar job (by name basename), or None.

    Uses the SQL ``query`` path (portable across iris CLI versions) — the older
    ``job list --json`` flag is not present on every iris build. State 3 == RUNNING.
    """
    out = run_iris(
        "query",
        f"SELECT job_id, state FROM jobs WHERE job_id LIKE '%/{JOB_NAME}'",
        "-f", "csv", check=False,
    )
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        job_id, state = parts[0], parts[1].lower()
        if job_id.rsplit("/", 1)[-1] == JOB_NAME and state in ("3", "running"):
            return {"job_id": job_id, "name": job_id, "state": "RUNNING"}
    return None


def ingress_host_from_logs(job_id: str) -> str | None:
    """Parse the ``INGRESS_HOST   https://...`` banner from the job's logs."""
    out = run_iris("job", "logs", job_id, "--tail", "--max-lines", "300", check=False)
    # Logs accumulate across re-submissions under the same job_id; take the most
    # recent banner (logs come back oldest-first), not a stale earlier one.
    matches = re.findall(r"INGRESS_HOST\s+(https://\S+)", out)
    return matches[-1] if matches else None


def stop_ingress_job(job_id: str) -> None:
    """Stop our own ingress job. Refuses to touch anything else (guardrail)."""
    if job_id.rsplit("/", 1)[-1] != JOB_NAME:
        raise SystemExit(f"refusing to stop non-ingress job {job_id}")
    log(f"[lifecycle] stopping existing ingress job {job_id}")
    run_iris("job", "stop", job_id, check=False)


def launch(url: str, token: str, api_key: str, region: str) -> str:
    """Submit the CPU-only, non-preemptible ingress job; return its job_id."""
    args = [
        "job", "run",
        "--cpu", "2", "--memory", "2GB", "--disk", "8GB",
        "--priority", "interactive", "--no-preemptible",
        *(["--region", region] if region else []),
        "--job-name", JOB_NAME,
        "-e", "IRIS_INGRESS_API_KEY", api_key,
        "-e", "PINGGY_URL", url,
        "-e", "PINGGY_TOKEN", token,
        "--no-wait",
        "--", "python", ENTRYPOINT,
    ]
    out = run_iris(*args, cwd=str(_REPO_ROOT), timeout=600)
    job_ids = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("/")]
    if not job_ids:
        raise SystemExit(f"could not parse job_id from `iris job run` output:\n{out[-800:]}")
    return job_ids[-1]


def wait_for_serving(job_id: str, timeout: int = 420) -> str:
    """Poll job logs until the INGRESS_HOST banner appears; fail on terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        host = ingress_host_from_logs(job_id)
        if host:
            return host
        summary = run_iris("job", "summary", job_id, check=False)
        if re.search(r"State:\s+(failed|terminated|dead|error|cancelled)", summary, re.I):
            raise SystemExit(f"job {job_id} went terminal before serving:\n{summary[-800:]}")
        time.sleep(15)
    raise SystemExit(f"job {job_id} did not print INGRESS_HOST within {timeout}s.")


def launch_and_verify(api_key: str, region: str, bank: str, attempts: int = 2) -> tuple[str, str]:
    """Pick a free pair, launch, wait, and G2-verify; retry on a fresh pair if G2 fails."""
    tried: set[str] = set()
    last = ""
    for _ in range(attempts):
        url, token = pick_free_pair(bank, exclude=tried)
        tried.add(url)
        job_id = launch(url, token, api_key, region)
        log(f"[deploy] launched {job_id} on {url}; waiting for it to serve...")
        host = wait_for_serving(job_id)
        # The pinggy banner prints as soon as the reverse tunnel registers, but
        # the public edge A record takes 1-3 min to propagate on a fresh bind;
        # wait for real forwarding before G2 or it false-fails with 000.
        if not is_public_healthy(host, tolerance=240):
            log(f"[deploy] {url} banner printed but public /healthz never forwarded.")
        ok, detail = g2_verify(host, api_key)
        if ok:
            log(f"[deploy] G2 OK {detail}")
            return host, job_id
        last = str(detail)
        log(f"[deploy] G2 FAILED on {url}: {detail}; tearing down and trying another pair.")
        stop_ingress_job(job_id)
        time.sleep(5)
    raise SystemExit(f"could not stand up a healthy ingress after {attempts} pairs; last G2={last}")


# --------------------------------------------------------------------------- #
# Subcommands.
# --------------------------------------------------------------------------- #
def cmd_deploy(args) -> int:
    api_key = require_key()
    job = find_running_job()
    if job:
        job_id = job["job_id"]
        host = ingress_host_from_logs(job_id)
        if host and healthz_ok(host):
            log(f"[deploy] existing job {job_id} already healthy.")
            print(f"INGRESS_HOST={host} JOB_ID={job_id}")
            return 0
        log(f"[deploy] existing job {job_id} running but unhealthy; replacing.")
        stop_ingress_job(job_id)
        time.sleep(5)
    host, job_id = launch_and_verify(api_key, args.region, args.bank)
    print(f"INGRESS_HOST={host} JOB_ID={job_id}")
    return 0


def cmd_ensure(args) -> int:
    api_key = require_key()
    job = find_running_job()
    if job:
        job_id = job["job_id"]
        host = ingress_host_from_logs(job_id)
        # Primary liveness = the iris job is RUNNING. The worker supervisor
        # keeps the sidecar + pinggy tunnel up (restarts the tunnel on drop;
        # fails the job if the sidecar dies), so a RUNNING job means the
        # service is up. The Mac-side pinggy probe is flaky here (ISP DNS
        # poison + pinggy edge migrations that blip for a minute or two), so
        # it is NOT a redeploy trigger on its own — only a LONG sustained
        # outage justifies nuking a running job.
        if host and is_public_healthy(host, tolerance=180):
            print(f"INGRESS_HOST={host} JOB_ID={job_id} STATUS=healthy")
            return 0
        log(f"[ensure] job {job_id} RUNNING but public /healthz down >180s; redeploying.")
        stop_ingress_job(job_id)
        time.sleep(5)
    else:
        log("[ensure] no running ingress job; deploying.")
    host, job_id = launch_and_verify(api_key, args.region, args.bank)
    print(f"INGRESS_HOST={host} JOB_ID={job_id} STATUS=redeployed")
    return 3  # non-zero: it had to act


def cmd_status(args) -> int:
    job = find_running_job()
    if not job:
        print("JOB_ID=none STATE=absent INGRESS_HOST=none HEALTHZ=n/a")
        return 1
    job_id = job["job_id"]
    state = job.get("state", "?")
    host = ingress_host_from_logs(job_id) or "unknown"
    healthy = host != "unknown" and is_public_healthy(host, tolerance=30)
    print(f"JOB_ID={job_id} STATE={state} INGRESS_HOST={host} HEALTHZ={'200' if healthy else '000'}")
    return 0 if healthy else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    for name in ("deploy", "ensure", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("--region", default=DEFAULT_REGION)
        sp.add_argument("--bank", default=DEFAULT_BANK)
    args = p.parse_args()
    cmd = args.cmd or "deploy"
    return {"deploy": cmd_deploy, "ensure": cmd_ensure, "status": cmd_status}[cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
