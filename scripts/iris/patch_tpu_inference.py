#!/usr/bin/env python3
"""Apply runtime patches to the iris worker's pip-installed tpu-inference.

Invoked from the launcher's bash bootstrap *after* ``uv sync`` and
*before* the workload starts. Each patch is idempotent (skipped if
already applied) and prints a one-line status so the iris-job log makes
the applied state obvious.

Patches currently applied
-------------------------

* **hbm_usage_bytes non-addressable device skip** (tpu_inference/utils.py)

  ``hbm_usage_bytes()`` iterates over every device in the JAX mesh and
  calls ``device.memory_stats()``. On multi-host TPU slices >v6e-8
  (v6e-16 = 4 hosts, v6e-32 = 8 hosts) each host can only address its
  4 local chips; calling ``memory_stats()`` on a non-addressable device
  raises ``jax.errors.JaxRuntimeError: INVALID_ARGUMENT: MemoryStats is
  only supported for addressable PjRt devices.``

  The patch adds a ``is_addressable`` guard to the non-Ray branch of
  the iteration so non-local chips are skipped instead of raising.
  This matches the Ray branch's behavior (which already short-circuits
  after the first successful device).
"""

from __future__ import annotations

import sys
from pathlib import Path


PATCHES: list[tuple[str, str, str, str]] = [
    # (relative_path, old_text, new_text, label)
    (
        "tpu_inference/utils.py",
        "    else:\n"
        "        for device in devices:\n"
        "            hbm_used = device.memory_stats()[\"bytes_in_use\"]\n"
        "            hbm_limit = device.memory_stats()[\"bytes_limit\"]\n"
        "            usage.append((hbm_used, hbm_limit))",
        "    else:\n"
        "        for device in devices:\n"
        "            # ot-agent patch (patch_tpu_inference.py):\n"
        "            # skip non-addressable devices on multi-host slices >v6e-8.\n"
        "            if not getattr(device, \"is_addressable\", True):\n"
        "                continue\n"
        "            try:\n"
        "                hbm_used = device.memory_stats()[\"bytes_in_use\"]\n"
        "                hbm_limit = device.memory_stats()[\"bytes_limit\"]\n"
        "            except Exception:\n"
        "                continue\n"
        "            usage.append((hbm_used, hbm_limit))",
        "hbm_usage_bytes: skip non-addressable devices on multi-host slices",
    ),
    (
        "tpu_inference/layers/common/utils.py",
        "    def _put(t):\n"
        "        multihost_backend = envs.TPU_MULTIHOST_BACKEND\n"
        "        # If we are not in multi-host setup, or the tensor is not fully addressable,\n"
        "        # we can use jax.device_put directly.\n"
        "        if multihost_backend != \"ray\" or (isinstance(t, jax.Array)\n"
        "                                          and not t.is_fully_addressable):\n"
        "            if layout is not None:\n"
        "                return jax.device_put(t, Format(layout, sharding))\n"
        "            else:\n"
        "                return jax.device_put(t, sharding)\n"
        "\n"
        "        # NOTE: at here, num_global_devices != num_local_devices\n"
        "        # meaning we are in multi-host setup. Each host will run the same process\n"
        "        # and each process only need to handle the devices accessible to this host.\n"
        "        ctx = nullcontext() if source_mesh is None else jax.set_mesh(\n"
        "            source_mesh)\n"
        "        # `t[i]` needs to be operated in the same mesh as `t`, which is provided as\n"
        "        # `source_mesh`.\n"
        "        with ctx:\n"
        "            global_array = jax.make_array_from_callback(\n"
        "                t.shape, sharding, lambda index: t[index])\n"
        "        if layout is not None:\n"
        "            dst_mesh = sharding.mesh\n"
        "            with jax.set_mesh(dst_mesh):\n"
        "                global_array = jax.device_put(global_array,\n"
        "                                              Format(layout, sharding))\n"
        "        return global_array",
        "    def _put(t):\n"
        "        multihost_backend = envs.TPU_MULTIHOST_BACKEND\n"
        "        # ot-agent patch (patch_tpu_inference.py): the upstream non-ray branch\n"
        "        # is single-host only -- on a multi-process job it cannot place a\n"
        "        # process-local host array onto the global mesh, and replicated\n"
        "        # weights also assert byte-equality (per-process FP8 dequant diverges).\n"
        "        # Route multi-process placement through make_array_from_callback (the\n"
        "        # ray primitive), broadcasting rank 0's copy of numeric weights first\n"
        "        # so replicated copies agree. broadcast_one_to_all returns a\n"
        "        # process-local numpy array, so it must also go through\n"
        "        # make_array_from_callback (not plain device_put). Extended dtypes\n"
        "        # (PRNGKeys) are seeded identically and cannot be broadcast, so they\n"
        "        # skip the broadcast and go straight to make_array_from_callback.\n"
        "        if multihost_backend == \"ray\":\n"
        "            if isinstance(t, jax.Array) and not t.is_fully_addressable:\n"
        "                if layout is not None:\n"
        "                    return jax.device_put(t, Format(layout, sharding))\n"
        "                return jax.device_put(t, sharding)\n"
        "            ctx = nullcontext() if source_mesh is None else jax.set_mesh(\n"
        "                source_mesh)\n"
        "            with ctx:\n"
        "                global_array = jax.make_array_from_callback(\n"
        "                    t.shape, sharding, lambda index: t[index])\n"
        "            if layout is not None:\n"
        "                with jax.set_mesh(sharding.mesh):\n"
        "                    global_array = jax.device_put(global_array,\n"
        "                                                  Format(layout, sharding))\n"
        "            return global_array\n"
        "        if jax.process_count() == 1:\n"
        "            if layout is not None:\n"
        "                return jax.device_put(t, Format(layout, sharding))\n"
        "            return jax.device_put(t, sharding)\n"
        "        if isinstance(t, jax.Array) and not t.is_fully_addressable:\n"
        "            if layout is not None:\n"
        "                return jax.device_put(t, Format(layout, sharding))\n"
        "            return jax.device_put(t, sharding)\n"
        "        if not jax.dtypes.issubdtype(t.dtype, jax.dtypes.extended):\n"
        "            from jax.experimental import multihost_utils as _ot_mhu\n"
        "            t = _ot_mhu.broadcast_one_to_all(t)\n"
        "        global_array = jax.make_array_from_callback(\n"
        "            t.shape, sharding, lambda index: t[index])\n"
        "        if layout is not None:\n"
        "            with jax.set_mesh(sharding.mesh):\n"
        "                global_array = jax.device_put(global_array,\n"
        "                                              Format(layout, sharding))\n"
        "        return global_array",
        "general_device_put: multi-process placement via make_array_from_callback "
        "+ rank-0 broadcast (FP8 device_put fix)",
    ),
]


def _site_packages_root() -> Path:
    """Return ``site-packages/`` for the current Python's environment.

    Prefer the venv at /app/.venv (iris worker convention); fall back
    to whichever directory hosts the running interpreter's site-packages.
    """
    candidate = Path("/app/.venv/lib/python3.12/site-packages")
    if candidate.is_dir():
        return candidate
    # Fallback: derive from sys.executable.
    for path in sys.path:
        p = Path(path)
        if p.name == "site-packages":
            return p
    raise RuntimeError(
        "could not locate site-packages on PYTHONPATH; aborting patch"
    )


def _apply_one(site_pkg: Path, rel_path: str, old: str, new: str, label: str) -> str:
    """Return a status string describing what the patch did."""
    target = site_pkg / rel_path
    if not target.is_file():
        return f"SKIP (file not found: {target}): {label}"
    src = target.read_text()
    if new in src:
        return f"ALREADY-PATCHED: {label}"
    if old not in src:
        return f"WARN (patch site not found, file may have drifted): {label}"
    target.write_text(src.replace(old, new))
    return f"APPLIED: {label}"


def main() -> int:
    site_pkg = _site_packages_root()
    print(f"[tpu-inference-patch] site-packages = {site_pkg}", flush=True)
    for rel_path, old, new, label in PATCHES:
        status = _apply_one(site_pkg, rel_path, old, new, label)
        print(f"[tpu-inference-patch] {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
