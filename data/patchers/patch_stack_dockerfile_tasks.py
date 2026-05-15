#!/usr/bin/env python3
"""
exp_rpt_stack-dockerfile patcher (filter-only).

Bug: 10/10 sampled v1 traces failed before the agent ran. Two failure
modes (~50/50):

  - **PATH_GLOB**: harness fails copying assets because the
    ``environment/Dockerfile`` references files that aren't in the
    extracted task dir (e.g. ``COPY ./entrypoint.sh ...``,
    ``COPY ./environment/conf/msmtprc ...``,
    ``COPY ./environment/package*.json ./``).
  - **BUILD_FAILED**: ``docker build`` itself fails:
      * Inaccessible images: ``pull access denied for evernym/dockerbase``,
        ``manifest for openjdk:8-jdk-alpine not found``,
        ``pull access denied for nodesource/vivid-base``.
      * Broken apt-get: ``python-qt4`` (Python 2, dropped in Ubuntu 22.04+),
        ``libgl1-mesa-glx`` (renamed to ``libgl1-mesa-dri``).

These are unfixable mechanically — we can't conjure deleted images or
invent missing assets. The right move is to filter.

Filter passes (in order):

  1. **PATH_GLOB**: scan ``environment/Dockerfile`` for ``COPY``/``ADD``
     source paths. Drop URL-only ``ADD`` lines, ``--from=`` build-stage
     references, and absolute paths (these are usually generated inside
     the build, e.g. ``COPY /usr/src/...``). For every relative source
     path, check the path exists under the task directory. If any path
     is missing (or a glob pattern matches zero files), drop the task.

  2. **BASE IMAGE**: extract ``FROM <image>`` references. For each
     non-``scratch`` / non-build-stage / non-private-registry reference,
     do a Docker Hub HEAD check. Cache results in
     ``/tmp/dockerhub_cache.json``. Drop on 404 (image / tag does not
     exist). Treat 5xx, network errors, and rate-limits as "kept" (don't
     drop on ambiguous status).

  3. **APT-GET**: scan for known-removed Debian/Ubuntu packages
     (``python-qt4``, ``libgl1-mesa-glx``, ``python-pip`` for old
     distros). Light filter — drops only the most obviously-broken
     installs.

Pure filter — surviving tasks are not modified.

Idempotent: dropping is a destructive op (``shutil.rmtree``); re-running
on a partially-filtered tree is safe (already-dropped tasks aren't
present).

Usage::

    python data/patchers/patch_stack_dockerfile_tasks.py \\
        --root /tmp/stack-dockerfile-extracted \\
        [--dry-run] [--limit N] [--no-online-check]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --------------------------------------------------------------------------- #
# Pass 1: PATH_GLOB
# --------------------------------------------------------------------------- #

# Match COPY / ADD lines. Captures the part after the verb. We deliberately
# do NOT try to be fully docker-shell-grammar-correct: we just want the
# source path arguments. Multi-line continuations (`\`) are joined first.
_COPY_ADD_RE = re.compile(r"^\s*(COPY|ADD)\s+(.+)$", re.IGNORECASE | re.MULTILINE)

# Common option flags that precede the source list (e.g. --chown=user:group,
# --from=builder, --chmod=755). We strip any leading `--*=*` tokens.
_DASH_OPT_RE = re.compile(r"^--\S+=\S+$")

# URL detector for ADD ... <url> ... — these don't need to exist on disk.
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _join_continuations(text: str) -> str:
    """Join `\\\\\\n` line continuations so each Dockerfile instruction is one line."""
    return re.sub(r"\\\s*\n", " ", text)


def _strip_inline_comment(line: str) -> str:
    # Dockerfile comments only at start of line (after whitespace).
    return line.split("#", 1)[0] if line.lstrip().startswith("#") else line


def _split_args(arg_str: str) -> list[str]:
    """
    Split a Docker COPY/ADD argument string into tokens.

    Handles JSON-array form (``["src1","src2","dst"]``) and plain
    whitespace-separated form. Strips quotes from individual tokens.
    """
    s = arg_str.strip()
    if s.startswith("["):
        # Try JSON-array first.
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except Exception:
            pass
    # Split on whitespace, honoring simple double-quoted strings.
    tokens: list[str] = []
    cur = []
    in_q = False
    quote_char = ""
    for ch in s:
        if in_q:
            if ch == quote_char:
                in_q = False
            else:
                cur.append(ch)
        else:
            if ch in ('"', "'"):
                in_q = True
                quote_char = ch
            elif ch.isspace():
                if cur:
                    tokens.append("".join(cur))
                    cur = []
            else:
                cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    return tokens


def _has_glob(s: str) -> bool:
    return any(c in s for c in "*?[")


def find_copy_add_sources(dockerfile_text: str) -> list[str]:
    """Return the list of *source* path arguments from COPY/ADD lines.

    Excludes:
      - URLs (``https://`` ADD)
      - ``--from=`` build-stage refs (those are stage-internal, not host)
      - The destination (last token of each instruction)

    Returns a deduped list, preserving order of first appearance.
    """
    text = _join_continuations(dockerfile_text)
    seen: set[str] = set()
    out: list[str] = []
    for m in _COPY_ADD_RE.finditer(text):
        verb = m.group(1).upper()
        rest = _strip_inline_comment(m.group(2)).strip()
        if not rest:
            continue
        tokens = _split_args(rest)
        # Strip leading option flags (--from=, --chown=, --chmod=, --link).
        i = 0
        from_stage = False
        while i < len(tokens) and (
            _DASH_OPT_RE.match(tokens[i]) or tokens[i].startswith("--")
        ):
            if tokens[i].startswith("--from="):
                from_stage = True
            i += 1
        if from_stage:
            # COPY --from=<stage> references the named stage's filesystem,
            # not the build context — no host file required.
            continue
        srcs_and_dst = tokens[i:]
        if len(srcs_and_dst) < 2:
            # Malformed; skip silently.
            continue
        # Last token is destination; everything before is source(s).
        sources = srcs_and_dst[:-1]
        for src in sources:
            if verb == "ADD" and _URL_RE.match(src):
                continue  # remote URL
            if src in seen:
                continue
            seen.add(src)
            out.append(src)
    return out


def path_glob_passes(task_dir: Path) -> tuple[bool, str]:
    """Return ``(ok, reason)``. ``ok=False`` means drop. ``reason`` is the
    first missing path or glob (for reporting)."""
    df = task_dir / "environment" / "Dockerfile"
    if not df.is_file():
        return False, "<no Dockerfile>"
    try:
        text = df.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"<read error: {e}>"

    # The Docker build context for the harness is the task's
    # ``environment/`` directory (the harness COPYs that into the image).
    # We accept paths under the task root OR under environment/, since
    # task-level files (e.g. tests/) are sometimes referenced too.
    env_dir = task_dir / "environment"

    sources = find_copy_add_sources(text)
    for src in sources:
        # Skip absolute paths that begin with /. These usually come from
        # multi-stage builds where an earlier stage put the file in a
        # known location. Without a --from= flag this is broken Docker
        # syntax, but it's not our job to flag that here — it'll fail in
        # build, not in path-resolution.
        if src.startswith("/"):
            continue
        # Strip ``./`` prefix.
        rel = src
        while rel.startswith("./"):
            rel = rel[2:]
        # Variable references (e.g. ``$CNS_BUILD_ARCHIVE``) — skip
        # checking, since the harness sets these up at build time. If the
        # ARG isn't set the build will fail downstream, but that's a
        # different filter.
        if "$" in rel:
            continue
        # Ignore "." (means whole context — always exists by definition).
        if rel in ("", "."):
            continue

        # Resolve relative to the task root; the harness uses task root
        # as the build context for COPYs since environment/Dockerfile
        # references like ``./environment/foo`` are common in this
        # corpus.
        candidates = [
            task_dir / rel,
            env_dir / rel,
        ]
        # Strip any leading ``environment/`` to handle cases where the
        # Dockerfile uses paths relative to the parent dir.
        if rel.startswith("environment/"):
            candidates.append(task_dir / rel[len("environment/") :])

        if _has_glob(rel):
            # At least one match must exist under one of the candidate roots.
            matched = False
            for base in [task_dir, env_dir]:
                # Use Path.glob with the leftover relative pattern.
                pattern = rel
                # Path.glob doesn't accept absolute patterns; rel is relative.
                try:
                    if any(base.glob(pattern)):
                        matched = True
                        break
                except (ValueError, OSError):
                    pass
            if not matched:
                return False, f"glob:{src}"
        else:
            if not any(c.exists() for c in candidates):
                return False, f"miss:{src}"
    return True, ""


# --------------------------------------------------------------------------- #
# Pass 2: BASE IMAGE
# --------------------------------------------------------------------------- #

# FROM <image>[:tag][@digest] [AS stage]   — case-insensitive
_FROM_RE = re.compile(
    r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+\S+)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Known-private / non-Docker-Hub registries we should skip checking.
_PRIVATE_REGISTRY_PREFIXES = (
    "gcr.io/",
    "us.gcr.io/",
    "eu.gcr.io/",
    "asia.gcr.io/",
    "k8s.gcr.io/",
    "registry.k8s.io/",
    "ghcr.io/",
    "quay.io/",
    "mcr.microsoft.com/",
    "public.ecr.aws/",
    "registry-1.docker.io/",
    "registry.gitlab.com/",
    "docker.elastic.co/",
    "docker.pkg.github.com/",
    "registry.access.redhat.com/",
    "registry.redhat.io/",
)


def _parse_image_ref(ref: str) -> tuple[str, str, str] | None:
    """Parse ``[registry/]org/image[:tag][@digest]`` for Docker Hub.

    Returns ``(namespace, repo, tag)`` for Docker Hub references, or
    ``None`` for non-Hub / unparseable refs.

    Examples:
        ubuntu:18.04 -> ("library", "ubuntu", "18.04")
        node:12-alpine -> ("library", "node", "12-alpine")
        balenalib/orangepi-plus2-alpine:edge-run
            -> ("balenalib", "orangepi-plus2-alpine", "edge-run")
        gcr.io/foo/bar -> None (handled by registry-prefix check)
    """
    # Drop digest.
    if "@" in ref:
        ref = ref.split("@", 1)[0]
    # If a registry path is present it'll have a "." or ":" before the
    # first "/". Docker treats these as non-Hub registries.
    parts = ref.split("/")
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        return None  # non-Hub registry
    # Now ref is either ``image[:tag]`` or ``namespace/image[:tag]``.
    if len(parts) == 1:
        ns = "library"
        rest = parts[0]
    elif len(parts) == 2:
        ns = parts[0]
        rest = parts[1]
    else:
        # 3+ slashes, but no registry prefix -> unusual; skip.
        return None
    if ":" in rest:
        repo, tag = rest.rsplit(":", 1)
    else:
        repo, tag = rest, "latest"
    return ns, repo, tag


def find_from_images(dockerfile_text: str) -> list[str]:
    """Return distinct external base-image refs (excludes build-stage refs)."""
    text = _join_continuations(dockerfile_text)
    # First pass: collect declared stage names (FROM x AS <stage>).
    stages = set()
    for line in text.splitlines():
        m = re.match(
            r"^\s*FROM\s+(?:--platform=\S+\s+)?\S+\s+AS\s+(\S+)\s*$",
            line,
            re.IGNORECASE,
        )
        if m:
            stages.add(m.group(1).lower())
    refs: list[str] = []
    seen: set[str] = set()
    for m in _FROM_RE.finditer(text):
        ref = m.group(1).strip()
        if not ref:
            continue
        # Variable ref — can't check.
        if "$" in ref or ref.startswith("${"):
            continue
        # ``scratch`` is a Docker built-in.
        if ref.lower() == "scratch":
            continue
        # Stage-name back-reference (FROM builder).
        if ref.lower() in stages:
            continue
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    return refs


# Cache file — survives across runs.
_CACHE_PATH = Path("/tmp/dockerhub_cache.json")


def _load_cache() -> dict[str, str]:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(cache))
    except Exception:
        pass


# Status codes:
#   "ok"       — image+tag verified to exist
#   "missing"  — 404 (drop)
#   "ambiguous"— anything else (network error, 5xx, rate-limit, skipped)
def check_image_exists(ref: str, cache: dict[str, str], timeout: float = 8.0) -> str:
    if ref in cache:
        return cache[ref]
    # Skip private registries.
    for prefix in _PRIVATE_REGISTRY_PREFIXES:
        if ref.startswith(prefix):
            cache[ref] = "ambiguous"
            return "ambiguous"
    parsed = _parse_image_ref(ref)
    if parsed is None:
        cache[ref] = "ambiguous"
        return "ambiguous"
    ns, repo, tag = parsed
    url = f"https://hub.docker.com/v2/repositories/{ns}/{repo}/tags/{tag}/"
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "patcher/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            if 200 <= status < 300:
                cache[ref] = "ok"
                return "ok"
            cache[ref] = "ambiguous"
            return "ambiguous"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            cache[ref] = "missing"
            return "missing"
        if e.code == 429:
            # Rate limited — back off briefly and treat as ambiguous.
            time.sleep(2.0)
            cache[ref] = "ambiguous"
            return "ambiguous"
        cache[ref] = "ambiguous"
        return "ambiguous"
    except (urllib.error.URLError, TimeoutError, OSError):
        cache[ref] = "ambiguous"
        return "ambiguous"


def base_image_passes(
    task_dir: Path,
    cache: dict[str, str],
    online: bool,
) -> tuple[bool, str]:
    df = task_dir / "environment" / "Dockerfile"
    try:
        text = df.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"<read error: {e}>"
    if not online:
        return True, ""
    for ref in find_from_images(text):
        status = check_image_exists(ref, cache)
        if status == "missing":
            return False, f"missing:{ref}"
    return True, ""


# --------------------------------------------------------------------------- #
# Pre-warm cache: collect all unique image refs across the corpus and HEAD-
# check them concurrently. This is *much* faster than per-task serial checks
# (10000 tasks × ~1s/task = 2.5h serial → ~2 min with 32 parallel workers and
# only ~1500 unique image refs).
# --------------------------------------------------------------------------- #

def collect_unique_images(task_dirs: list[Path]) -> list[str]:
    seen: set[str] = set()
    for td in task_dirs:
        df = td / "environment" / "Dockerfile"
        if not df.is_file():
            continue
        try:
            text = df.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for ref in find_from_images(text):
            seen.add(ref)
    return sorted(seen)


def prewarm_image_cache(
    images: list[str],
    cache: dict[str, str],
    workers: int = 32,
    timeout: float = 6.0,
    progress_every: int = 100,
) -> None:
    """HEAD-check every unseen image in parallel. Mutates ``cache`` in place."""
    todo = [img for img in images if img not in cache]
    if not todo:
        return
    print(f"[prewarm] {len(todo)} unique image refs to check (workers={workers})")
    done_count = 0

    def _job(ref: str) -> tuple[str, str]:
        return ref, check_image_exists(ref, {}, timeout=timeout)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_job, ref): ref for ref in todo}
        for fut in as_completed(futures):
            try:
                ref, status = fut.result()
                cache[ref] = status
            except Exception as e:
                ref = futures[fut]
                cache[ref] = "ambiguous"
                print(f"  ! prewarm {ref}: {e}", file=sys.stderr)
            done_count += 1
            if done_count % progress_every == 0 or done_count == len(todo):
                print(f"[prewarm] {done_count}/{len(todo)}", flush=True)
    _save_cache(cache)


# --------------------------------------------------------------------------- #
# Pass 3: APT-GET sanity (light)
# --------------------------------------------------------------------------- #

# Packages we know cause apt-get install to fail on modern Ubuntu/Debian.
# This is intentionally a small list — we err on the side of keeping
# tasks unless the breakage is near-certain.
_BAD_APT_PACKAGES = {
    # Python 2 stack — removed in Ubuntu 22.04 LTS+.
    "python-qt4",
    "python-qt4-gl",
    "python-qt4-dev",
    "python-pyqt4",
    "python-pyqt5",  # debian-style py2 build, removed
    # Mesa rename in modern distros (libgl1-mesa-glx → libgl1-mesa-dri).
    "libgl1-mesa-glx",
    # Old pip / setuptools transitions.
    "python-pip",
    "python-setuptools",
    "python-wheel",
}

_APT_INSTALL_RE = re.compile(
    r"\bapt-get\s+(?:-[a-zA-Z]+\s+)*install\b([^\n&|;]*)",
    re.IGNORECASE,
)


def apt_get_passes(task_dir: Path) -> tuple[bool, str]:
    df = task_dir / "environment" / "Dockerfile"
    try:
        text = df.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"<read error: {e}>"
    text = _join_continuations(text)
    # Only check tasks that target modern Ubuntu/Debian as their base.
    # Old base images (ubuntu:18.04, debian:9) historically had these
    # packages — flagging them would over-drop. We only flag when the
    # FROM image is ubuntu:22.04+ / debian:12+ / ``ubuntu`` / ``debian``
    # (latest), where the breakage is real.
    from_images = [r.lower() for r in find_from_images(text)]

    def _is_modern(img: str) -> bool:
        # ``ubuntu`` / ``debian`` (no tag → latest, modern).
        if img in ("ubuntu", "debian"):
            return True
        # ubuntu:22.04, ubuntu:24.04, ubuntu:noble, ubuntu:jammy.
        if img.startswith("ubuntu:"):
            tag = img.split(":", 1)[1]
            if tag in ("latest", "noble", "jammy", "lunar", "mantic", "oracular"):
                return True
            try:
                year = int(tag.split(".")[0])
                if year >= 22:
                    return True
            except (ValueError, IndexError):
                pass
        if img.startswith("debian:"):
            tag = img.split(":", 1)[1]
            if tag in ("latest", "bookworm", "trixie"):
                return True
            try:
                ver = int(tag.split("-")[0])
                if ver >= 12:
                    return True
            except (ValueError, IndexError):
                pass
        return False

    if not any(_is_modern(i) for i in from_images):
        return True, ""

    for m in _APT_INSTALL_RE.finditer(text):
        pkgs_blob = m.group(1)
        # Tokenize: drop options and version-pins.
        for tok in pkgs_blob.split():
            tok = tok.strip().rstrip("\\")
            if not tok or tok.startswith("-"):
                continue
            # Strip version pin foo=1.2.3
            pkg = tok.split("=", 1)[0]
            if pkg in _BAD_APT_PACKAGES:
                return False, f"apt:{pkg}"
    return True, ""


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def filter_one_task(
    task_dir: Path,
    cache: dict[str, str],
    online: bool,
) -> tuple[str, str, str]:
    """Run all three filters in order. Return ``(verdict, pass, reason)``.

    verdict ∈ {"keep", "drop"}; ``pass`` is the failing pass name on drop
    (``""`` when keep); ``reason`` is a short string for reporting.

    Also captures the FROM image of the task (first ref) for the report.
    """
    df = task_dir / "environment" / "Dockerfile"
    base_image = ""
    if df.is_file():
        try:
            text = df.read_text(encoding="utf-8", errors="replace")
            imgs = find_from_images(text)
            if imgs:
                base_image = imgs[0]
        except Exception:
            pass

    ok, why = path_glob_passes(task_dir)
    if not ok:
        return "drop", "path_glob", f"{why}|from={base_image}"

    ok, why = base_image_passes(task_dir, cache, online)
    if not ok:
        return "drop", "image_404", f"{why}|from={base_image}"

    ok, why = apt_get_passes(task_dir)
    if not ok:
        return "drop", "apt_broken", f"{why}|from={base_image}"

    return "keep", "", f"from={base_image}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="Directory of extracted task folders")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't actually delete dropped task dirs")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-online-check", action="store_true",
                   help="Skip Docker Hub HEAD checks")
    p.add_argument("--examples", type=int, default=10)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not task_dirs:
        print(f"No task dirs under {root}", file=sys.stderr)
        return 2
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    cache = _load_cache()
    online = not args.no_online_check

    # Pre-warm the image-check cache: parallel HEAD checks across all unique
    # image refs in the corpus. This turns 10K serial network calls into
    # ~1.5K parallel calls.
    if online:
        all_images = collect_unique_images(task_dirs)
        print(f"[prewarm] discovered {len(all_images)} unique base-image refs")
        prewarm_image_cache(all_images, cache, workers=32, timeout=6.0)

    counts: dict[str, int] = {"keep": 0, "drop_path_glob": 0,
                              "drop_image_404": 0, "drop_apt_broken": 0}
    examples: dict[str, list[str]] = {}
    # Track base-image distribution for kept vs each drop bucket.
    from collections import Counter
    base_counter_keep: Counter = Counter()
    base_counter_drop: Counter = Counter()

    total = len(task_dirs)
    for i, td in enumerate(task_dirs, 1):
        verdict, fail_pass, reason = filter_one_task(td, cache, online)
        # Extract from= from reason (last segment).
        base_image = ""
        if "from=" in reason:
            base_image = reason.split("from=", 1)[1]

        if verdict == "keep":
            counts["keep"] += 1
            if base_image:
                base_counter_keep[base_image] += 1
            examples.setdefault("keep", [])
            if len(examples["keep"]) < args.examples:
                examples["keep"].append(f"{td.name} ({base_image})")
        else:
            key = f"drop_{fail_pass}"
            counts[key] += 1
            if base_image:
                base_counter_drop[base_image] += 1
            examples.setdefault(key, [])
            if len(examples[key]) < args.examples:
                examples[key].append(f"{td.name} :: {reason}")
            if not args.dry_run:
                try:
                    shutil.rmtree(td)
                except Exception as e:
                    print(f"  ! rmtree {td.name}: {e}", file=sys.stderr)

        if i % 250 == 0 or i == total:
            sample = ", ".join(f"{k}={v}" for k, v in counts.items())
            print(f"[{i}/{total}] {sample}", flush=True)
            # Persist cache periodically so a Ctrl-C doesn't lose progress.
            if online and i % 1000 == 0:
                _save_cache(cache)

    if online:
        _save_cache(cache)

    print(f"\nDone. {total} task dirs processed (dry_run={args.dry_run}).")
    kept = counts["keep"]
    yield_pct = kept / total * 100 if total else 0.0
    print(f"  Yield: {kept}/{total} = {yield_pct:.1f}%")
    for k in ("keep", "drop_path_glob", "drop_image_404", "drop_apt_broken"):
        v = counts[k]
        pct = v / total * 100 if total else 0.0
        print(f"  {k:<20}: {v:>5} ({pct:5.1f}%)")
        for name in examples.get(k, [])[: args.examples]:
            print(f"      {name}")

    print("\nTop 10 base images in KEPT set:")
    for img, n in base_counter_keep.most_common(10):
        print(f"  {n:>5}  {img}")
    print("\nTop 10 base images in DROPPED set:")
    for img, n in base_counter_drop.most_common(10):
        print(f"  {n:>5}  {img}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
