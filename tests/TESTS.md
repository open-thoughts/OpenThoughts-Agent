# TESTS.md — repo test-suite catalog / awareness ledger

**Last updated:** 2026-07-12 (test-suite reorg, Stages 0/B/C — see
`notes/ot-agent/test-suite-reorg-plan.md`).

One `tests/` tree holds every genuine, repo-level automated test, organized by
subsystem. The infra-dependency axis is carried by **pytest markers** (registered
in `pyproject.toml [tool.pytest.ini_options]`), not by directories. Default
(unmarked) = CI-safe unit.

- **CI (green lane):** `pytest tests/ -q` → **361 passed, 2 skipped** (363 collected).
- **Awareness:** `pytest --collect-only -q` lists every test (testpaths=["tests"]).
- **Markers:** `gpu · cluster · daytona · hf · supabase · net · slow · integration`.
  Off-target infra-marked tests auto-SKIP (not ERROR) via `tests/conftest.py`.

---

## CI-safe unit suites (in `tests/`, unmarked)

### `tests/hpc/` (16 test files)
`test_export_literal_threading, test_fetch_daemon_watchdog, test_ingress_parity_harness,
test_ingress_sidecar, test_ingress_wiring, test_iris_launcher_args,
test_launch_utils_collision, test_literal_correlator, test_literal_proxy,
test_literal_rescue, test_literal_to_sft, test_resolve_rl_repo_dir,
test_resume_determinism, test_resume_manager, test_rl_chain_overshoot_guard,
test_snapshot_manager` — pure unit; mock Daytona/Harbor in-process. No move.

### `tests/eval/` (6 test files)
`test_iris_thinking_kwarg, test_listener_agent_kwargs, test_per_model_agent_kwargs,
test_per_model_timeout_multiplier, test_vllm_parallelism_sizing` (pre-existing)
**+ `test_model_registry_resolve`
(migrated Stage B, reshaped script→pytest)**.

---

## Migrated into `tests/` (Stage B — 2026-07-12)

| New home | Old path | Tests | Marker | Import fix |
|---|---|---|---|---|
| `tests/data/nemotron_gym/test_adapter.py` | `data/nemotron_gym/tests/test_adapter.py` | 34 | — (CI-safe) | 14× `from ..adapter`/`..verifiers`/`..converters.*` → absolute `data.nemotron_gym.*`; removed a dead `df = … if False` line |
| `tests/data/nl2bash/test_sampler.py` | `data/nl2bash_sampled_verified/tests/test_sampler.py` | 9 | — | none (self-contained) |
| `tests/data/nl2bash/test_validation.py` | `…/tests/test_validation.py` | 7 | — | none (bodies are `pass`; harmless sys.path line left) |
| `tests/data/nl2bash/test_output_normalization.py` | `…/tests/test_output_normalization.py` | 13 | — | removed unused `subprocess`/`Path` imports |
| `tests/data/negotiation/test_generate.py` | `data/negotiation/test_generate.py` | 28 (2 skip) | — | `ROOT parents[2]→[3]`; dropped redundant sys.path hack (data is installed) |
| `tests/eval/test_model_registry_resolve.py` | `eval/tests/test_model_registry_resolve.py` | 1 (aggregate) | — | **RESHAPED** print/`sys.exit` script → single pytest test; **monkeypatch-guards** the `unified_eval_listener` module globals (`_CLUSTER_CONFIG`, `resolve_base_model_name`, `_BASELINE_MODEL_*`) so its in-process mutation can't leak into sibling eval tests |

## QUARANTINED — broken, NOT migrated (needs a separate fix pass)

| File (left in place) | Why | Action |
|---|---|---|
| `data/nl2bash_sampled_verified/tests/test_failure_categorization.py` (11 tests) | Imports a top-level `analyze_failures` module that **does not exist anywhere in the repo** → **ERRORs on collection**. A broken file can't collect; migrating it into `tests/` would wedge the CI suite. | Left OUTSIDE `tests/` (so CI never collects it). Fix: restore/point at the real `categorize_failure` source, then migrate to `tests/data/nl2bash/`. |

---

## Ad-hoc CLI / repro scripts — RENAMED off the `test_` namespace (Stage C — 2026-07-12)

These are `argparse`/`main()`/print-based smoke harnesses (live LLM/S3/cluster/HF),
**not pytest**. Renamed in place off the `test_` prefix so pytest never mis-collects
them (they remain runnable manual tools next to their dataset/area).

| Old (`test_*`) | New |
|---|---|
| `data/all_puzzles/test_generate.py` | `data/all_puzzles/generate_smoke.py` |
| `data/perturbed_docker/test_generate.py` | `data/perturbed_docker/generate_smoke.py` |
| `data/ta_rl_tasks/test_generate.py` | `data/ta_rl_tasks/generate_smoke.py` |
| `data/taskmaster2/test_generate.py` | `data/taskmaster2/generate_smoke.py` |
| `data/trace_hints/test_generate.py` | `data/trace_hints/generate_smoke.py` |
| `data/bespoke/test_converted_tasks.py` | `data/bespoke/check_converted_tasks.py` |
| `data/freelancer_filtering/test_filters.py` | `data/freelancer_filtering/filters_smoke.py` |
| `rl/hpc/test_hpc.py` | `rl/hpc/hpc_smoke.py` (updated refs: `rl/hpc/setup.sh`, `rl/hpc/launch_utils/pre_download_dataset.py` docstring) |
| `scripts/datagen/test_reasoning_content.py` | `scripts/datagen/reasoning_content_smoke.py` |
| `scripts/beam/test_beta9_sandbox.py` | `scripts/beam/beta9_sandbox_smoke.py` |
| `scripts/s3/test_bucket.py` | `scripts/s3/bucket_smoke.py` |

Per operator: these are RELOCATE+RENAME only (incl. the 3 borderline
`rl/hpc/hpc_smoke.py`, `freelancer_filtering/filters_smoke.py`,
`bespoke/check_converted_tasks.py`) — NOT upgraded into real pytest tests.

---

## Terminal-Bench task assets — DO NOT move; excluded from collection (Group D — 10 files)

Carry the `terminal-bench-canary GUID` header + assert in-container paths
(`/app`, `/logs/verifier`). Copied into the Daytona/Docker container by the Harbor
harness and run *there* — benchmark ground-truth, not repo tests. Excluded via
`norecursedirs = ["few-shots", "templates", …]` (basename match) + `testpaths=["tests"]`.

- `data/self_instruct/few-shots/{build-cython-ext,chess-best-move,configure-git-webserver,fix-code-vulnerability,polyglot-c-py,qemu-alpine-ssh,qemu-startup,regex-log}/tests/test_outputs.py` (8)
- `data/self_instruct/few-shots/hello-world/tests/test_state.py`
- `data/frontiersmith/templates/test_state.py`

## Naming false-positives (NOT tests — leave alone)

- `data/patchers/patch_*stack_pytest_*_tasks.py` (5) — dataset patcher scripts (`pytest` substring).
- `scripts/analysis/filter_latest_episodes.py` — `latest_` → `test_` substring.
