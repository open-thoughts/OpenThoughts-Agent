#!/usr/bin/env python
"""Stage-0 submit-surface parity harness for the eval-listener unification plan.

Companion: notes/ot-agent/eval_listener_unification_plan.md (Stage 0).

The sibling `resolve_parity_harness.py` captures the resolved SERVE outputs
(per-model env / conda_env / agent_kwargs triple) over the full (model x cluster)
matrix -- the gate for the model-config-decoupling refactor. THIS harness captures
the SUBMIT surface -- the resolved ListenerConfig + the per-(cluster, preset, model)
`sbatch` argv + the full EVAL_* env dict that `submit_eval` would export -- which is
what the listener-unification stages can perturb:

  * Stage 1 (thin wrapper): does routing through `hpc.launch` change the env or the
    forwarded flags the listener sees? Gate G2.
  * Stage 4 (merge eval/clusters into HPC): does moving the cluster config change
    the resolved ListenerConfig fields or the sbatch argv/env? Gate G3.

Design (mirrors resolve_parity_harness.py discipline):
  * Uses the REAL listener code -- `parse_args`, `build_config`, `EvalListener`,
    `get_vllm_env_overrides`, `get_conda_env_override`, `SbatchParams`, and
    `submit_eval` are all imported + called, NOT reimplemented. A later refactor
    flips `--check` FAIL->PASS without editing this harness.
  * Submission is INTERCEPTED: `uel._run` is monkeypatched to record (cmd, env)
    and return a fake "Submitted batch job" success; `submit_eval` is called with
    `dry_run=False` so it exercises the REAL cmd-building + to_env() path
    (the `dry_run=True` branch short-circuits BEFORE building the cmd). The DB
    block is forced to skip (get_supabase_client raises -> try/except -> db_job_id
    stays None) so no Pending row is written.
  * Determinism: run_tag is pinned via `run_tag_override="HARNESS_RUN_TAG"`; the
    USER env var + upload_username are pinned; passthru env vars
    (SWEAGENT_CONFIG / EVAL_USE_GLM5_PROXY / EVAL_JOBS_DIR / EVAL_SLURM_JOB_NAME)
    are cleared; resolve_base_model_name (Supabase) is stubbed to None (hermetic).
  * Absence is load-bearing: the captured env dict records EXACTLY what to_env()
    returns (a missing EVAL_VLLM_MAX_MODEL_LEN stays missing so the sbatch
    `:-32768` default applies) -- a key present in one snapshot and absent in the
    other is a diff.
  * Memoization: _BASELINE_MODEL_CONFIGS / _BASELINE_MODEL_PATTERNS / _CLUSTER_CONFIG
    are module globals; reset between cluster cells so cluster N doesn't reuse
    cluster N-1's configs (same footgun as resolve_parity_harness.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# --- repo root on sys.path + deterministic env BEFORE importing the listener ---
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Pin DCFT (some config paths / expandvars reference it) to a stable sentinel so the
# snapshot is location-independent (same trick as resolve_parity_harness.py).
os.environ["DCFT"] = "/__DCFT__"
# Pin USER so upload_username (getpass.getuser fallback) is deterministic.
os.environ["USER"] = "harness_user"
# Clear passthru / override env vars that submit_eval reads from os.environ so the
# captured env is deterministic regardless of the shell the harness runs in.
for _k in ("SWEAGENT_CONFIG", "EVAL_USE_GLM5_PROXY", "EVAL_JOBS_DIR",
           "EVAL_SLURM_JOB_NAME", "HF_HUB_CACHE"):
    os.environ.pop(_k, None)

import eval.unified_eval_listener as uel  # noqa: E402


# --- Representative matrix -----------------------------------------------------
# 2 clusters (proxy off / on; different gpus_per_node + arch) x 2 presets
# (different n_concurrent + datasets) x 3 models exercising distinct resolution
# paths (default serve, TP override via size, qwen3.5-style divergence). Small but
# spans the axes Stages 1 + 4 can perturb; the full (model x cluster) SERVE matrix
# is already covered by resolve_parity_harness.py.
_CLUSTERS = [
    {"name": "leonardo", "yaml": "eval/clusters/leonardo.yaml"},
    {"name": "jupiter", "yaml": "eval/clusters/jupiter.yaml"},
]
_PRESETS = ["tb2", "v2"]
# (label, hf_model_name) -- chosen to exercise: 8B default, 32B TP-override, unknown.
_MODELS = [
    ("qwen3-8b", "Qwen/Qwen3-8B"),
    ("qwen3-32b", "Qwen/Qwen3-32B"),
    ("unknown-default", "DCAgent/harness-probe-model"),
]

# ListenerConfig fields that drive submission and could be perturbed by Stage 4's
# cluster-config merge. Excludes nondeterministic fields (log_file path+timestamp,
# upload_username, priority_models contents, cluster_config raw dict).
_LC_FIELDS = [
    "datasets", "sbatch_script", "n_concurrent", "n_attempts", "gpu_memory_util",
    "error_threshold", "vllm_max_retries", "agent_parser", "slurm_time",
    "slurm_partition", "slurm_account", "tp_size", "dp_size", "agent_name",
    "config_yaml", "timeout_multiplier", "use_model_registry", "model_registry",
    "hardware_profile", "conda_env", "dp_sbatch_script", "agent_envs",
    "pre_download", "batch_size", "check_hf_exists", "force_reeval",
    "require_priority_list", "priority_mode",
]


def _reset_memo() -> None:
    """Reset the listener's module-global memoization so cluster N is independent."""
    uel._BASELINE_MODEL_CONFIGS = None
    uel._BASELINE_MODEL_PATTERNS = None
    uel._CLUSTER_CONFIG = None
    uel._BASE_MODEL_NAME_CACHE.clear()


def _build_listener_config(cluster_yaml: str, preset: str, tmp_log_dir: Path) -> object:
    """parse_args + build_config for one (cluster, preset) cell -> ListenerConfig."""
    # parse_args() reads sys.argv directly (no argv param); set it temporarily.
    argv = [
        "--cluster-config", str(_REPO_ROOT / cluster_yaml),
        "--preset", preset,
        "--once", "--dry-run",
        "--log-dir", str(tmp_log_dir),
        "--upload-username", "harness_user",
    ]
    orig_argv = sys.argv
    sys.argv = ["unified_eval_listener.py"] + argv
    try:
        args = uel.parse_args()
    finally:
        sys.argv = orig_argv
    # build_config derives log_file from --log-dir + a timestamp; pin to a fixed
    # path so EvalLogger.__init__'s set_log_file() is deterministic + side-effect-free.
    args.log_file = str(tmp_log_dir / f"{preset}_harness.log")
    return uel.build_config(args)


def _build_sbatch_params(config: object) -> "uel.SbatchParams":
    """Mirror EvalListener.run_iteration's SbatchParams construction (L3415-3437)."""
    return uel.SbatchParams(
        n_concurrent=config.n_concurrent,
        n_attempts=config.n_attempts,
        gpu_memory_util=config.gpu_memory_util,
        error_threshold=config.error_threshold,
        vllm_max_retries=config.vllm_max_retries,
        agent_parser=config.agent_parser,
        slurm_time=config.slurm_time,
        agent_kwargs=config.agent_kwargs,
        agent_name=config.agent_name,
        slurm_partition=config.slurm_partition,
        slurm_account=config.slurm_account,
        tp_size=config.tp_size,
        dp_size=config.dp_size,
        upload_username=config.upload_username,
        timeout_multiplier=config.timeout_multiplier,
        config_yaml=config.config_yaml,
        max_output_tokens=config.max_output_tokens,
        auto_snapshot=config.auto_snapshot,
        agent_envs=config.agent_envs,
        pinggy_url=config.pinggy_url,
        pinggy_token=config.pinggy_token,
    )


def _load_model_configs(config: object) -> dict:
    """Mirror run_iteration's baseline-configs load (registry default-on)."""
    if config.use_model_registry:
        profile = None if (config.hardware_profile in (None, "default")) else config.hardware_profile
        return uel.load_model_registry(config.model_registry, profile)
    if config.baseline_model_configs:
        return uel.load_baseline_model_configs(config.baseline_model_configs)
    return {}


def _capture_submit(model_hf: str, dataset_hf: str, sbatch_script: str,
                    sbatch_params: "uel.SbatchParams", config: object,
                    listener: "uel.EvalListener", configs: dict) -> dict:
    """Call submit_eval with _run intercepted; return {sbatch_argv, eval_env}."""
    vllm_overrides = uel.get_vllm_env_overrides(model_hf, configs)
    model_conda_env = uel.get_conda_env_override(model_hf, configs) or config.conda_env
    model_agent_kwargs = listener._resolve_agent_kwargs(model_hf, configs)
    model_tm = listener._resolve_timeout_multiplier(model_hf, configs)

    # Mirror the extra_env assembly from run_iteration (L3620-3632).
    extra_env: dict = {}
    if model_tm != config.timeout_multiplier:
        extra_env["EVAL_TIMEOUT_MULTIPLIER"] = str(model_tm)
    if model_agent_kwargs != list(config.agent_kwargs):
        extra_env["EVAL_AGENT_KWARGS"] = "\n".join(model_agent_kwargs)

    captured: dict = {}

    def _fake_run(cmd, env=None):
        captured["sbatch_argv"] = list(cmd)
        # env here is the listener's intended env_vars dict (NOT the merged
        # os.environ copy -- _run does that merge itself; we capture pre-merge).
        captured["eval_env"] = dict(env) if env else {}
        return (0, "Submitted batch job 0000000")

    # Force the DB block to skip cleanly: get_supabase_client raising makes the
    # try/except in submit_eval fall through with db_job_id=None (no Pending row,
    # no EVAL_DB_JOB_ID in env). Patched on the source module so the in-function
    # `from database.unified_db.utils import get_supabase_client` re-bind catches it.
    import database.unified_db.utils as dbu  # noqa: E402

    orig_run = uel._run
    orig_get_client = getattr(dbu, "get_supabase_client", None)
    uel._run = _fake_run
    dbu.get_supabase_client = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("harness: DB disabled"))
    try:
        uel.submit_eval(
            model_hf, dataset_hf, None, sbatch_script,
            sbatch_params=sbatch_params,
            dry_run=False,  # MUST be False: dry_run=True short-circuits before cmd build
            upload_username=config.upload_username,
            timeout_multiplier=config.timeout_multiplier,
            vllm_overrides=vllm_overrides or None,
            conda_env=model_conda_env,
            extra_env=extra_env or None,
            run_tag_override="HARNESS_RUN_TAG",  # pin the timestamp-derived job_name
            dp_nodes=0, nodelist=None,
            skip_gpu_request=False, sbatch_model_override=None,
        )
    finally:
        uel._run = orig_run
        if orig_get_client is not None:
            dbu.get_supabase_client = orig_get_client

    return captured


def build_snapshot() -> dict:
    """Resolve the submit surface for every (cluster, preset, model) cell."""
    # Hermetic: no Supabase base-model lookups (offline-safe; identical old/new).
    uel.resolve_base_model_name = lambda hf_model: None  # type: ignore[assignment]

    snapshot: dict = {"_meta": {}, "cells": {}}

    with tempfile.TemporaryDirectory(prefix="listener_harness_") as tmp:
        tmp_log_dir = Path(tmp)
        for cluster in _CLUSTERS:
            _reset_memo()
            cname = cluster["name"]
            config = _build_listener_config(cluster["yaml"], _PRESETS[0], tmp_log_dir)
            # ListenerConfig is preset-dependent only in the fields we re-derive
            # per preset below; reuse the EvalListener for the per-model resolvers
            # (its methods don't depend on preset once configs are loaded).
            # NOTE: build_config sets _CLUSTER_CONFIG + CONDA_ENV_PATHS as side effects.
            uel.EvalListener(config)
            _load_model_configs(config)

            # Capture the ListenerConfig once per cluster (preset-independent for
            # the cluster-derived fields Stage 4 risks; we still capture per-preset
            # below for the preset-derived fields).
            for preset in _PRESETS:
                _reset_memo()
                # Re-build per preset so preset-derived fields (n_concurrent,
                # config_yaml, datasets, agent_envs) are correct.
                pcfg = _build_listener_config(cluster["yaml"], preset, tmp_log_dir)
                # Re-instantiate so _resolve_* see this preset's config.
                listener_p = uel.EvalListener(pcfg)
                configs_p = _load_model_configs(pcfg)
                sbatch_params = _build_sbatch_params(pcfg)
                actual_sbatch = pcfg.sbatch_script

                lc_view = {f: getattr(pcfg, f) for f in _LC_FIELDS}

                cell_key = f"{cname}|{preset}"
                snapshot["cells"][cell_key] = {
                    "listener_config": lc_view,
                    "models": {},
                }
                for label, model_hf in _MODELS:
                    # tb2 preset -> DCAgent2/terminal_bench_2; v2 -> DCAgent/dev_set_v2.
                    dataset_hf = pcfg.datasets[0] if pcfg.datasets else ""
                    captured = _capture_submit(
                        model_hf, dataset_hf, actual_sbatch, sbatch_params,
                        pcfg, listener_p, configs_p,
                    )
                    snapshot["cells"][cell_key]["models"][label] = {
                        "hf_model": model_hf,
                        "dataset": dataset_hf,
                        "sbatch_argv": captured.get("sbatch_argv", []),
                        "eval_env": captured.get("eval_env", {}),
                    }

    snapshot["_meta"] = {
        "n_clusters": len(_CLUSTERS),
        "n_presets": len(_PRESETS),
        "n_models": len(_MODELS),
        "n_cells": len(_CLUSTERS) * len(_PRESETS),
        "lc_fields": _LC_FIELDS,
    }
    return snapshot


def _canonical_json(obj: dict) -> str:
    """Stable, byte-deterministic JSON (sorted keys, fixed separators)."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _diff_snapshots(golden: dict, current: dict) -> list:
    """Per-(cell, model, field/key) byte diffs between two snapshots."""
    diffs: list = []
    g_cells, c_cells = golden.get("cells", {}), current.get("cells", {})
    for cell in sorted(set(g_cells) | set(c_cells)):
        g, c = g_cells.get(cell), c_cells.get(cell)
        if g is None:
            diffs.append(f"[{cell}] present in CURRENT, absent in GOLDEN")
            continue
        if c is None:
            diffs.append(f"[{cell}] present in GOLDEN, absent in CURRENT")
            continue
        # ListenerConfig
        glc, clc = g.get("listener_config", {}), c.get("listener_config", {})
        for k in sorted(set(glc) | set(clc)):
            if glc.get(k) != clc.get(k):
                diffs.append(f"[{cell}] listener_config.{k}: GOLDEN={glc.get(k)!r} CURRENT={clc.get(k)!r}")
        # Per-model submit artifacts
        gm, cm = g.get("models", {}), c.get("models", {})
        for m in sorted(set(gm) | set(cm)):
            gmm, cmm = gm.get(m), cm.get(m)
            if gmm is None or cmm is None:
                diffs.append(f"[{cell}] model {m}: GOLDEN={'<abs>' if gmm is None else 'present'} CURRENT={'<abs>' if cmm is None else 'present'}")
                continue
            for fk in ("hf_model", "dataset"):
                if gmm.get(fk) != cmm.get(fk):
                    diffs.append(f"[{cell}] {m}.{fk}: GOLDEN={gmm.get(fk)!r} CURRENT={cmm.get(fk)!r}")
            # sbatch argv (list equality)
            if gmm.get("sbatch_argv") != cmm.get("sbatch_argv"):
                diffs.append(f"[{cell}] {m}.sbatch_argv:\n    GOLDEN ={gmm.get('sbatch_argv')!r}\n    CURRENT={cmm.get('sbatch_argv')!r}")
            # eval_env (dict, key-absence-sensitive)
            ge, ce = gmm.get("eval_env", {}), cmm.get("eval_env", {})
            for k in sorted(set(ge) | set(ce)):
                gv = ge.get(k, "<ABSENT>")
                cv = ce.get(k, "<ABSENT>")
                if gv != cv:
                    diffs.append(f"[{cell}] {m}.eval_env[{k}]: GOLDEN={gv!r} CURRENT={cv!r}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", metavar="PATH", help="resolve all cells and WRITE the golden JSON here")
    ap.add_argument("--check", metavar="GOLDEN", help="resolve all cells and DIFF against this golden; nonzero on any diff")
    args = ap.parse_args()

    if not args.write and not args.check:
        ap.error("one of --write / --check is required")

    snapshot = build_snapshot()
    meta = snapshot["_meta"]
    print(
        f"== listener submit-surface harness: {meta['n_cells']} cells "
        f"({meta['n_clusters']} clusters x {meta['n_presets']} presets), "
        f"{meta['n_models']} models/cell ==",
        file=sys.stderr,
    )

    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_canonical_json(snapshot))
        print(f"WROTE golden: {out} ({meta['n_cells']} cells)", file=sys.stderr)
        return 0

    golden = json.loads(Path(args.check).read_text())
    diffs = _diff_snapshots(golden, snapshot)
    if diffs:
        print(f"PARITY FAIL: {len(diffs)} byte-diff(s) vs {args.check}:", file=sys.stderr)
        for d in diffs:
            print(f"  {d}", file=sys.stderr)
        return 1
    print(f"PARITY OK: byte-identical vs {args.check} across {meta['n_cells']} cells.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
